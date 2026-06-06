import os
import io
import json
import re
import base64
import hashlib
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Iterator, Pattern
from PIL import Image
import fitz
import pdfplumber
from openai import OpenAI
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from sentence_transformers import SentenceTransformer

# OPENAI_API_KEY must be set in environment before importing this module
model = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.0)
str_parser = StrOutputParser()

# Global vector-store handles – populated by initialize_rag()
chroma_vs_1 = None
chroma_vs_2 = None
chroma_img_1 = None
chroma_img_2 = None
clip_model = None

# ─────────────────────────────────────────────────────────────
# INGESTION
# ─────────────────────────────────────────────────────────────

FOOTER_PATTERNS = [
    re.compile(r"The Consumer Product Safety Office \(CPSO\) safeguards consumer safety"),
    re.compile(r"with applicable safety standards\."),
    re.compile(r"The CPSO is an office of the Competition and Consumer Commission of Singapore"),
    re.compile(r"the Ministry of Trade and Industry\."),
    re.compile(r"Visit www\.consumerproductsafety\.gov\.sg for more information\."),
]
PAGE_NUMBER_PATTERN = re.compile(r"^\d{1,3}$")


def pdf_loader(pdf_path: str) -> List[Document]:
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    doc = fitz.open(pdf_path)
    for page_index, page in enumerate(pages):
        fitz_page = doc[page_index]
        image_list = fitz_page.get_images(full=True)
        page_images = []
        for img in image_list:
            xref = img[0]
            base_image = doc.extract_image(xref)
            base64_img = base64.b64encode(base_image["image"]).decode("utf-8")
            page_images.append(base64_img)
        page.metadata["raw_images"] = page_images
        page.metadata["has_visuals"] = bool(page_images)
    return pages


def is_footer_or_page_number(line: str) -> bool:
    s = line.strip()
    if not s:
        return True
    if PAGE_NUMBER_PATTERN.fullmatch(s):
        return True
    return any(p.search(s) for p in FOOTER_PATTERNS)


def split_by_section(
    docs: List[Document],
    heading_pattern: Pattern,
    metadata_key: str,
) -> List[Document]:
    chunks = []
    buffer = []
    current_title = None
    current_metadata = None
    for doc in docs:
        for line in doc.page_content.splitlines():
            stripped = line.strip()
            if is_footer_or_page_number(stripped):
                continue
            if heading_pattern.match(stripped):
                if buffer:
                    chunks.append(Document(
                        page_content="\n".join(buffer),
                        metadata={**current_metadata, metadata_key: current_title},
                    ))
                    buffer = []
                current_title = stripped
                current_metadata = doc.metadata
            elif current_title:
                buffer.append(line)
    if buffer:
        chunks.append(Document(
            page_content="\n".join(buffer),
            metadata={**current_metadata, metadata_key: current_title},
        ))
    return chunks


def splitting_section_pipeline(pdf_path: str, h1_pattern: Pattern, pdf_index: int) -> List[Document]:
    pages = pdf_loader(pdf_path)
    return split_by_section(pages, h1_pattern, "section")


def reconstruct_docs(section_docs: List[Document], subsection_docs: List[Document]) -> List[Document]:
    sections_with_subs = {
        doc.metadata.get("section")
        for doc in subsection_docs
        if "subsection" in doc.metadata
    }
    final_docs = list(subsection_docs)
    for sec_doc in section_docs:
        section_title = sec_doc.metadata.get("section")
        if section_title not in sections_with_subs:
            final_docs.append(Document(
                page_content=sec_doc.page_content,
                metadata={**sec_doc.metadata, "subsection": "-"},
            ))
    return sorted(
        final_docs,
        key=lambda x: (x.metadata.get("section", ""), x.metadata.get("subsection", "")),
    )


def filter_subsections(docs: List[Document]) -> List[Document]:
    filtered = []
    for doc in docs:
        section = doc.metadata.get("section", "")
        subsection = doc.metadata.get("subsection", "")
        sec_match = re.search(r"^(\d+)", section)
        sub_match = re.search(r"^(\d+)", subsection)
        if sec_match and sub_match and sec_match.group(1) != sub_match.group(1):
            continue
        if any(ch in subsection for ch in [";", ","]):
            continue
        if section.startswith("4.0") or section.startswith("5.0"):
            if not re.match(r"^\d+\.", subsection):
                continue
        if ".PDF" in subsection.upper() or "**" in subsection:
            continue
        filtered.append(doc)
    return filtered


def extract_table_docs(pdf_path: str, chunks: List[Document], source_label: str) -> List[Document]:
    raw_map: Dict[int, dict] = {}
    for chunk in chunks:
        pg = chunk.metadata.get("page")
        if pg is not None and chunk.metadata.get("section"):
            raw_map[pg] = {
                "section":    chunk.metadata.get("section", ""),
                "subsection": chunk.metadata.get("subsection", ""),
                "source":     chunk.metadata.get("source", pdf_path),
            }

    page_meta_map: Dict[int, dict] = {}
    if raw_map:
        first_page = min(raw_map.keys())
        for pg in range(first_page):
            page_meta_map[pg] = {"section": "Preliminary", "subsection": "-", "source": pdf_path}
        last_meta = raw_map[first_page]
        with pdfplumber.open(pdf_path) as _pdf:
            total_pages = len(_pdf.pages)
        for pg in range(first_page, total_pages):
            if pg in raw_map:
                last_meta = raw_map[pg]
            page_meta_map[pg] = last_meta.copy()

    table_docs = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            tables = page.extract_tables()
            if not tables:
                continue
            meta = page_meta_map.get(page_num, {"section": "", "subsection": "", "source": pdf_path})
            for table in tables:
                if not table or len(table) < 2:
                    continue
                header = table[0]
                rows = table[1:]
                md = "| " + " | ".join(str(c or "") for c in header) + " |\n"
                md += "| " + " | ".join("---" for _ in header) + " |\n"
                for row in rows:
                    md += "| " + " | ".join(str(c or "") for c in row) + " |\n"
                table_docs.append(Document(
                    page_content=md,
                    metadata={
                        "source":       meta["source"],
                        "section":      meta["section"],
                        "subsection":   meta["subsection"],
                        "page":         page_num,
                        "doc_type":     "table",
                        "source_label": source_label,
                    },
                ))
    print(f"{source_label}: {len(table_docs)} table documents extracted")
    return table_docs


def extract_heading_structure(
    collections_dict: Dict[str, Dict[str, Any]]
) -> Dict[str, Dict[str, List[str]]]:
    final_structure = {}
    for label, collection in collections_dict.items():
        structure: Dict[str, List[str]] = defaultdict(list)
        seen: Dict[str, set] = defaultdict(set)
        for metadata in collection.get("metadatas", []):
            h1 = metadata.get("section")
            h2 = metadata.get("subsection")
            if h1:
                if h2 and h2 != "-" and h2 not in seen[h1]:
                    structure[h1].append(h2)
                    seen[h1].add(h2)
                elif h1 not in structure:
                    structure[h1] = []
        final_structure[label] = dict(structure)
    return final_structure


def build_image_vector_store(chunks: List[Document], chroma_img, source_label: str) -> None:
    count = 0
    seen: set = set()
    for chunk in chunks:
        for idx, b64 in enumerate(chunk.metadata.get("raw_images", [])):
            fingerprint = hashlib.md5(b64.encode()).hexdigest()
            if fingerprint in seen:
                continue
            seen.add(fingerprint)
            try:
                pil_image = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
                if pil_image.width < 80 or pil_image.height < 80:
                    continue
                image_vector = clip_model.encode(pil_image).tolist()
                chroma_img._collection.add(
                    ids=[f"{source_label}_img_{count}"],
                    embeddings=[image_vector],
                    documents=[f"image {count} from {source_label}"],
                    metadatas=[{"source_label": source_label, "base64_image": b64}],
                )
                count += 1
            except Exception as e:
                print(f"Skipped image {idx} in chunk: {e}")
    print(f"{source_label}: {count} image vectors stored")


# ─────────────────────────────────────────────────────────────
# STATE DATACLASSES
# ─────────────────────────────────────────────────────────────

@dataclass
class WorkingMemory:
    compiled_query: str = ""
    route: str = ""
    target_doc: str = ""
    relevant_sections: List[str] = field(default_factory=list)
    relevant_subsections: List[str] = field(default_factory=list)
    raw_context: List[str] = field(default_factory=list)
    compiled_context: str = ""
    retrieved_docs: List[Document] = field(default_factory=list)
    retrieved_images: List[dict] = field(default_factory=list)

    def reset(self) -> None:
        self.compiled_query = ""
        self.route = ""
        self.target_doc = ""
        self.relevant_sections = []
        self.relevant_subsections = []
        self.raw_context = []
        self.compiled_context = ""
        self.retrieved_docs = []
        self.retrieved_images = []


@dataclass
class AssistantState:
    heading_structure: Dict[str, Dict[str, List[str]]]
    chat_history: List[BaseMessage] = field(default_factory=list)
    working_memory: WorkingMemory = field(default_factory=WorkingMemory)
    _doc_map: Dict[str, str] = field(default_factory=lambda: {
        "PDF_1": "heading structure 1",
        "PDF_2": "heading structure 2",
    })

    def reset(self) -> None:
        self.working_memory.reset()

    @property
    def input_query(self) -> Optional[str]:
        return self.chat_history[-1].content if self.chat_history else None

    @property
    def past_queries(self) -> List[str]:
        return [m.content for m in self.chat_history[:-1] if isinstance(m, HumanMessage)]

    def all_sections(self) -> str:
        sections = []
        for doc_key in self.heading_structure:
            sections.extend(self.heading_structure[doc_key].keys())
        return json.dumps(list(dict.fromkeys(sections)), indent=4, ensure_ascii=False)

    def all_subsections(self, sections: List[str]) -> str:
        subsections = []
        for doc_key in self.heading_structure:
            doc_sections = self.heading_structure[doc_key]
            for section in sections:
                if section in doc_sections:
                    subsections.extend(doc_sections[section])
        return json.dumps(list(dict.fromkeys(subsections)), indent=4, ensure_ascii=False)

    def allowed_subsections(self, section: str) -> List[str]:
        for doc_label in self.heading_structure:
            if section in self.heading_structure[doc_label]:
                return self.heading_structure[doc_label][section]
        return []

    def add_user_query(self, text: str) -> None:
        self.chat_history.append(HumanMessage(content=text))

    def add_ai_response(self, text: str) -> None:
        self.chat_history.append(AIMessage(content=text))


# ─────────────────────────────────────────────────────────────
# RAG CHAINS
# ─────────────────────────────────────────────────────────────

compiler_system_prompt = """
You are a specialist in the Singapore Consumer Product Safety (CPSR) framework.

Your task is to rewrite the user's query into a standalone retrieval query.

Context:
- CPSA+ Guidebook -> system usage
- CPSR Regulations -> legal requirements

Instructions:
- Rewrite informal, incomplete, or ungrammatical user queries into a clear and formal retrieval query.
- Use past queries only when the current query contains references, omissions, or follow-up phrases
  that cannot be understood on its own (e.g., "that requirement", "the fee", "renewal", "how long").
- If the current query is already complete, do not add unnecessary past context.
- Include domain keywords when relevant (e.g., SDoC, Controlled Goods, CPSA+, CoC, Technical File, LOA).
- If the query is about fees or deadlines, specify the exact context.
- Output ONE clear sentence only.
- Do NOT answer the question; only rewrite it as a retrieval query.

Examples:
User query: hello, I dont know how to use SDOC application?
Rewritten query: What is the SDoC application process in the CPSA+ system?

User query: how to view paymen ah?
Rewritten query: How can I view ongoing payments in the CPSA+ system?

User query: why we need download list of CGs and LOA
Rewritten query: What is the purpose of downloading the list of controlled goods and the LOA in CPSA+?

User query: key regulation of CPSR is what
Rewritten query: What are the key regulations under the Consumer Protection (Safety Requirements) Regulations?

User query: Medium Risk Controlled Goods???
Rewritten query: What are the safety standards and requirements for medium-risk controlled goods?

User query: what about the fee?
Past queries: How to apply for SDoC?
Rewritten query: What is the fee for SDoC application in the CPSA+ system?

User query: how long does it take?
Past queries: Tell me about SDoC renewal
Rewritten query: What is the processing time for SDoC renewal in the CPSA+ system?
"""

query_compiler_prompt = ChatPromptTemplate.from_messages([
    ("system", compiler_system_prompt),
    ("user", "\nQuery history: {past_queries}\n\nCurrent query: {input_query}\n"),
])

query_compiler_chain = (
    {
        "past_queries": lambda s: " | ".join(s.past_queries),
        "input_query":  lambda s: s.input_query,
    }
    | query_compiler_prompt
    | model
    | str_parser
)

# ── Router ────────────────────────────────────────────────────

router_system_prompt = """
You are an expert router for Singapore Consumer Product Safety documents.
Your task is to decide which document(s) should be used to answer the retrieval query.

Available documents:

- PDF_1: CPSA+ Guidebook for Registered Suppliers
  Use this for procedural or system-usage questions, such as:
  logging into and create RS account, dashboard usage,
  Controlled Goods Status(CoC)/Controlled Goods Status(SDoC)/Ongoing Allocated Certification Number(ACN),
  application steps, payment steps,
  draft applications, LOA download, and controlled goods list.

- PDF_2: Consumer Protection (Safety Requirements) Regulations
  Use this for legal, regulatory, registration, or technical questions, such as:
  controlled goods definitions, applicable safety standards, risk levels,
  additional requirements for registration of controlled goods,
  parties that need to apply for registration,
  registration of supplier of controlled goods,
  registration of controlled goods,
  Certificate of Conformity (CoC),
  modifications to registered controlled goods,
  Technical File, SAFETY Mark, and exemption-related requirements.

Routing rules:
- Output 'PDF_1' if the query is mainly about system procedures, user actions, navigation, or application steps in CPSA+.
- Output 'PDF_2' if the query is mainly about legal requirements, registration rules, standards, definitions, risk levels, CoC requirements, or regulatory conditions.
- Output 'BOTH' if the query clearly requires both procedural guidance and regulatory information.

- Output 'UNRELATED' if:
  * The query is NOT about Singapore Consumer Product Safety
  * OR it does not relate to CPSA+, controlled goods, CoC, SDoC, SAFETY Mark, or regulations
  * OR it is a general question (e.g., weather, coding, math, personal advice, etc.)
  * OR the intent cannot be answered using the provided documents

Output ONLY one of these exact strings:
PDF_1
PDF_2
BOTH
UNRELATED
"""

query_router_prompt = ChatPromptTemplate.from_messages([
    ("system", router_system_prompt),
    ("user", "{compiled_query}"),
])

query_router_chain = (
    {"compiled_query": lambda x: x.working_memory.compiled_query}
    | query_router_prompt
    | model
    | StrOutputParser()
)

# ── Analyzer 1 ────────────────────────────────────────────────

def parse_list_output(text: str) -> List[str]:
    text = text.strip()
    if not text or text.lower() in ["", "none", '""', "n/a"]:
        return []
    items = [item.strip() for item in text.split(",")] if "," in text else [line.strip() for line in text.split("\n")]
    return [item.strip().strip('"').strip("'").rstrip(".") for item in items if item.strip()]


list_parser = RunnableLambda(parse_list_output)


def normalize_route(route: str) -> Optional[str]:
    if not route:
        return None
    route = route.upper().replace(" ", "").replace("_", "")
    return {"PDF1": "PDF_1", "PDF2": "PDF_2", "BOTH": "BOTH", "UNRELATED": "UNRELATED"}.get(route)


def get_routed_sections(state: "AssistantState") -> List[str]:
    route = normalize_route(getattr(state.working_memory, "route", None))
    route_to_doc = {"PDF_1": "heading structure 1", "PDF_2": "heading structure 2"}
    if route == "UNRELATED":
        return []
    if route == "PDF_1":
        return list(state.heading_structure.get(route_to_doc["PDF_1"], {}).keys())
    if route == "PDF_2":
        return list(state.heading_structure.get(route_to_doc["PDF_2"], {}).keys())
    if route == "BOTH":
        pdf1 = state.heading_structure.get(route_to_doc["PDF_1"], {})
        pdf2 = state.heading_structure.get(route_to_doc["PDF_2"], {})
        return list(pdf1.keys()) + list(pdf2.keys())
    return []


def format_sections_for_prompt(state: "AssistantState") -> str:
    sections = get_routed_sections(state)
    return "\n".join(f"- {sec}" for sec in sections) if sections else ""


analyzer_1_system_prompt = """
You are a precision-focused Document Indexing Specialist for the Singapore Consumer Product Safety (CPSR) framework.
Your goal is to map a user's query to specific MAIN SECTION titles from a provided list.

### Selection Instructions:
1. **Keyword Priority**: If the query mentions 'SDoC', 'CoC', 'ACN', 'Safety Mark', or 'Technical File', select every section title containing those terms.
2. **Intent Mapping**:
   - For procedural/system usage (e.g., "how to pay", "where to click"), select sections from Document 1 (Guidebook).
   - For regulatory/legal requirements (e.g., "what is a controlled good", "safety standards"), select sections from Document 2 (Regulations).
3. **High Recall**: If a section is even "plausibly" relevant, include it. It is better to retrieve slightly more context than to miss the answer.
4. **Strict Output**: You must ONLY output the exact strings from the 'Available' list, separated by commas. Do not summarize or add conversational filler.

### Example:
Query: "How do I update my profile and check the status of my SDoC?"
Output: 5. RS Profile Page Overview, 6. Overview: "My Controlled Goods" Page, 6B. SDoC Registration
"""

analyzer_chain_1 = (
    {
        "query":    lambda s: s.working_memory.compiled_query,
        "sections": lambda s: format_sections_for_prompt(s),
        "route":    lambda s: s.working_memory.route,
    }
    | PromptTemplate.from_template(
        f"{analyzer_1_system_prompt}\n\n"
        "User Query: {query}\n"
        "Router Result: {route}\n\n"
        "Available Document Sections:\n{sections}\n\n"
        "Selected Relevant Sections (comma separated):"
    )
    | model
    | StrOutputParser()
    | list_parser
)

# ── Analyzer 2 ────────────────────────────────────────────────

def get_relevant_subsections(state: "AssistantState") -> str:
    relevant_sections = getattr(state.working_memory, "relevant_sections", None)
    if not relevant_sections:
        return ""
    return state.all_subsections(relevant_sections)


analyzer_2_system_prompt = """
You are a document navigation assistant.

Your task is to select ALL relevant SUBSECTIONS that help answer the user query.

### Instructions:

1. High Recall:
   - Include ALL subsections that are even moderately relevant
   - Prefer including more rather than missing useful content

2. Coverage:
   - If a section is selected, include its important subsections
   - Include both overview (e.g., status, dashboard) AND actions (e.g., apply, submit, renew)

3. Relevance:
   - Include subsections related to:
     - tracking / status
     - searching / viewing
     - application / submission
     - payment / renewal
   - These are all considered part of "managing Controlled Goods"

4. Do NOT include:
   - Completely unrelated sections
   - Duplicates

5. Output format:
   - ONLY comma-separated subsection titles
   - No explanation
"""

analyzer_chain_2 = (
    {
        "query":    lambda s: s.working_memory.compiled_query,
        "sections": lambda s: get_relevant_subsections(s),
    }
    | PromptTemplate.from_template(
        f"{analyzer_2_system_prompt}\n\n"
        "User Query: {query}\n\n"
        "Available Subsections:\n{sections}\n\n"
        "Relevant Subselections (comma separated):"
    )
    | model
    | StrOutputParser()
    | list_parser
)

# ── Parallel Retriever ────────────────────────────────────────

def get_target_doc_labels(state: "AssistantState") -> List[str]:
    route = getattr(state.working_memory, "route", "")
    if route == "PDF_1":
        return ["heading structure 1"]
    if route == "PDF_2":
        return ["heading structure 2"]
    if route == "BOTH":
        return ["heading structure 1", "heading structure 2"]
    return []


def get_vectorstore_by_label(doc_label: str):
    return {"heading structure 1": chroma_vs_1, "heading structure 2": chroma_vs_2}.get(doc_label)


def build_section_subsection_pairs(state: "AssistantState") -> List[Tuple[str, str, str]]:
    pairs = []
    target_labels = get_target_doc_labels(state)
    relevant_sections = state.working_memory.relevant_sections or []
    relevant_subsections = state.working_memory.relevant_subsections or []
    if not target_labels or not relevant_sections:
        return pairs
    for doc_label in target_labels:
        doc_structure = state.heading_structure.get(doc_label, {})
        for section in relevant_sections:
            if section not in doc_structure:
                continue
            allowed_subs = doc_structure[section]
            if allowed_subs:
                for subsection in relevant_subsections:
                    if subsection in allowed_subs:
                        pairs.append((doc_label, section, subsection))
            else:
                pairs.append((doc_label, section, "-"))
    return pairs


def build_dynamic_retrievers(state: "AssistantState", k: int = 3) -> Dict[str, Any]:
    retrievers = {}
    for i, (doc_label, section, subsection) in enumerate(build_section_subsection_pairs(state)):
        vectorstore = get_vectorstore_by_label(doc_label)
        if vectorstore is None:
            continue
        retrievers[f"{doc_label}__{section}__{subsection}__{i}"] = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": k,
                "filter": {"$and": [{"section": section}, {"subsection": subsection}]},
            },
        )
    return retrievers


def build_section_level_retrievers(state: "AssistantState", k: int = 3) -> Dict[str, Any]:
    retrievers = {}
    idx = 0
    for doc_label in get_target_doc_labels(state):
        vectorstore = get_vectorstore_by_label(doc_label)
        doc_structure = state.heading_structure.get(doc_label, {})
        if vectorstore is None:
            continue
        for section in (state.working_memory.relevant_sections or []):
            if section not in doc_structure:
                continue
            retrievers[f"{doc_label}__{section}__section_only__{idx}"] = vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": k, "filter": {"section": section}},
            )
            idx += 1
    return retrievers


def deduplicate_docs(docs: List[Document]) -> List[Document]:
    unique_docs, seen = [], set()
    for doc in docs:
        key = (
            doc.metadata.get("section", ""),
            doc.metadata.get("subsection", ""),
            doc.metadata.get("chunk_id", ""),
            doc.page_content[:120],
        )
        if key not in seen:
            seen.add(key)
            unique_docs.append(doc)
    return unique_docs


def run_parallel_retriever(state: "AssistantState", k: int = 3) -> List[dict]:
    query = state.working_memory.compiled_query
    if not query:
        state.working_memory.raw_context = []
        state.working_memory.retrieved_docs = []
        return []

    retrievers = build_dynamic_retrievers(state, k=k)
    if not retrievers:
        retrievers = build_section_level_retrievers(state, k=k)
    if not retrievers:
        state.working_memory.raw_context = []
        state.working_memory.retrieved_docs = []
        return []

    parallel_retriever_chain = (
        RunnableParallel(retrievers)
        | RunnableLambda(lambda results: [doc for result in results.values() for doc in result])
        | RunnableLambda(deduplicate_docs)
        | RunnableLambda(lambda docs: sorted(docs, key=lambda doc: doc.metadata.get("chunk_id", 10**9)))
    )

    retrieved_docs = parallel_retriever_chain.invoke(query)
    raw_context = [
        {"text": doc.page_content, "images": doc.metadata.get("raw_images", [])}
        for doc in retrieved_docs
    ]
    state.working_memory.retrieved_docs = retrieved_docs
    state.working_memory.raw_context = raw_context

    # Image retrieval via CLIP
    img_store_map = {"heading structure 1": chroma_img_1, "heading structure 2": chroma_img_2}
    retrieved_images = []
    for doc_label in get_target_doc_labels(state):
        img_vs = img_store_map.get(doc_label)
        if img_vs is None or img_vs._collection.count() == 0:
            continue
        query_vector = clip_model.encode(query).tolist()
        relevant_sections = state.working_memory.relevant_sections or []
        results = img_vs._collection.query(
            query_embeddings=[query_vector],
            n_results=min(2, img_vs._collection.count()),
            where={"section": {"$in": relevant_sections}} if relevant_sections else None,
            include=["metadatas", "distances"],
        )
        for meta in results["metadatas"][0]:
            b64 = meta.get("base64_image", "")
            if b64:
                retrieved_images.append({
                    "base64":  b64,
                    "section": meta.get("section", ""),
                    "source":  meta.get("source", ""),
                })
    state.working_memory.retrieved_images = retrieved_images
    return raw_context


# ── Context Compiler ──────────────────────────────────────────

def format_raw_context(state: "AssistantState") -> str:
    contexts = state.working_memory.raw_context
    if not contexts:
        return "No retrieved context available."
    formatted = []
    for i, chunk in enumerate(contexts, start=1):
        text = chunk.get("text", "")
        has_image = "Yes" if chunk.get("images") else "No"
        formatted.append(f"[Context {i}]\n{text}\n[Has Image: {has_image}]")
    return "\n\n".join(formatted)


context_compiler_prompt = PromptTemplate.from_template(
    """
You are a Context Compiler for a retrieval-based assistant on Singapore Consumer Product Safety documents.

Your task is to filter, compress, and organize the retrieved context into a clean, high-signal evidence summary.

Instructions:
- Use ONLY the retrieved contexts provided.
- Keep only information directly relevant to the query.
- Remove repetition, overlap, irrelevant details, and noise.
- Preserve important procedural, regulatory, and technical details.
- Do NOT answer the user directly.
- Do NOT add outside knowledge.
- If the context is insufficient, state that clearly.

Output format:
To (based on question), you can:-
- Key point 1:
  ... \\n
- Key point 2:
  ... \\n
- Key point 3:
  ... \\n

The Key point should paraphase based on related content and bold it.

Query:
{query}

Retrieved Contexts:
{contexts}
"""
)

context_compiler_chain = (
    {
        "query":    lambda s: s.working_memory.compiled_query,
        "contexts": lambda s: format_raw_context(s),
    }
    | context_compiler_prompt
    | model
    | str_parser
)


# ── Responder ─────────────────────────────────────────────────

def _multimodal_responder_fn(state: "AssistantState") -> AIMessage:
    client = OpenAI()
    images  = state.working_memory.retrieved_images or []
    context = state.working_memory.compiled_context or "No context available."
    query   = state.chat_history[-1].content if state.chat_history else ""

    messages = [{
        "role": "system",
        "content": (
            "You are a retrieval-based assistant for Singapore Consumer Product Safety documents.\n"
            "Answer ONLY from the provided context and images. Do not use outside knowledge.\n"
            "If an image is provided, describe what you see and incorporate it into your answer."
        ),
    }]

    for msg in state.chat_history[:-1]:
        if isinstance(msg, HumanMessage):
            messages.append({"role": "user",      "content": msg.content})
        elif isinstance(msg, AIMessage):
            messages.append({"role": "assistant", "content": msg.content})

    if images:
        content: List[dict] = [
            {"type": "text", "text": f"Retrieved Context:\n{context}"},
            {"type": "text", "text": f"\n{len(images)} relevant image(s) retrieved:"},
        ]
        for i, img in enumerate(images):
            content.append({"type": "text", "text": f"\n[Image {i+1} | Section: '{img['section']}' | Source: {img['source']}]"})
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img['base64']}", "detail": "high"}})
        content.append({"type": "text", "text": f"\nUser Question: {query}"})
        messages.append({"role": "user", "content": content})
    else:
        messages.append({"role": "user", "content": f"Retrieved Context:\n{context}\n\nUser Question: {query}"})

    resp = client.chat.completions.create(
        model="gpt-5.4",
        messages=messages,
        max_completion_tokens=1000,
        temperature=0.0,
    )
    return AIMessage(content=resp.choices[0].message.content)


responder_chain = RunnableLambda(_multimodal_responder_fn)


# ─────────────────────────────────────────────────────────────
# ASSISTANT
# ─────────────────────────────────────────────────────────────

class Assistant(Runnable):
    def __init__(self, model: ChatOpenAI, state: AssistantState):
        super().__init__()
        self.model = model
        self.state = state

    def stream(self, input: str, config: Optional[dict] = None) -> Iterator[str]:
        self.state.reset()
        self.state.add_user_query(input)

        yield "\n\n<sub>⚙️ Compiling query...</sub>\n\n"
        self.state.working_memory.compiled_query = query_compiler_chain.invoke(self.state)

        yield "\n\n<sub>⚙️ Routing query to the correct document...</sub>\n\n"
        self.state.working_memory.route = query_router_chain.invoke(self.state).strip()
        self.state.working_memory.target_doc = self.state.working_memory.route

        if self.state.working_memory.route == "UNRELATED":
            self.state.working_memory.relevant_sections = []
            self.state.working_memory.relevant_subsections = []
            self.state.working_memory.raw_context = []
            self.state.working_memory.compiled_context = (
                "The question is unrelated to the provided Singapore Consumer Product Safety documents."
            )
        else:
            yield "\n\n<sub>⚙️ Identifying relevant main sections...</sub>\n\n"
            self.state.working_memory.relevant_sections = analyzer_chain_1.invoke(self.state) or []

            if self.state.working_memory.relevant_sections:
                yield "\n\n<sub>⚙️ Identifying relevant subsections...</sub>\n\n"
                self.state.working_memory.relevant_subsections = analyzer_chain_2.invoke(self.state) or []

                yield "\n\n<sub>⚙️ Retrieving relevant chunks...</sub>\n\n"
                self.state.working_memory.raw_context = run_parallel_retriever(self.state)

                yield "\n\n<sub>💬 Finalizing retrieved evidence...</sub>\n\n"
                self.state.working_memory.compiled_context = context_compiler_chain.invoke(self.state)
            else:
                self.state.working_memory.relevant_subsections = []
                self.state.working_memory.raw_context = []
                self.state.working_memory.compiled_context = (
                    "No relevant information was found in the provided documents."
                )

        yield "<br>"

        n_imgs = len(self.state.working_memory.retrieved_images or [])
        if n_imgs:
            yield f"\n\n<sub>🖼️ Reasoning over {n_imgs} image(s)...</sub>\n\n"

        result = responder_chain.invoke(self.state)
        final_text = result.content if hasattr(result, "content") else str(result)

        for char in final_text:
            yield char

        if final_text.strip():
            self.state.add_ai_response(final_text.strip())

    def invoke(self, input: str, config: Optional[dict] = None, **kwargs: Any) -> AIMessage:
        for _ in self.stream(input, config=config):
            pass
        if self.state.chat_history and isinstance(self.state.chat_history[-1], AIMessage):
            return self.state.chat_history[-1]
        return AIMessage(content="")


# ─────────────────────────────────────────────────────────────
# INITIALIZATION
# ─────────────────────────────────────────────────────────────

def initialize_rag(
    pdf_path1: str = "CPSA+_Guidebook_for_Registered_Suppliers.pdf",
    pdf_path2: str = "Consumer_Protection_Safety_Requirements_Regulations.pdf",
    text_db_path: str = "./my_text_db",
    image_db_path: str = "./my_image_db",
    heading_cache: str = "./heading_structure.json",
) -> Assistant:
    """
    Build or reload the RAG pipeline.
    On the first run (no persisted stores) this ingests both PDFs, which takes several minutes.
    On subsequent runs it loads from disk instantly.
    """
    global chroma_vs_1, chroma_vs_2, chroma_img_1, chroma_img_2, clip_model

    print("Loading embedding models...")
    embedding_model = HuggingFaceEmbeddings(model_name="clip-ViT-B-32")
    clip_model = SentenceTransformer("clip-ViT-B-32")

    chroma_vs_1 = Chroma(
        collection_name="registered_suppliers_guidebook",
        embedding_function=embedding_model,
        persist_directory=text_db_path,
    )
    chroma_vs_2 = Chroma(
        collection_name="consumer_protection_regulations",
        embedding_function=embedding_model,
        persist_directory=text_db_path,
    )
    chroma_img_1 = Chroma(collection_name="image_vectors_pdf1_v2", persist_directory=image_db_path)
    chroma_img_2 = Chroma(collection_name="image_vectors_pdf2_v2", persist_directory=image_db_path)

    # Use cached heading structure + existing vector stores if available
    if os.path.exists(heading_cache) and chroma_vs_1._collection.count() > 0:
        print("Loading from existing vector stores...")
        with open(heading_cache, encoding="utf-8") as f:
            heading_structure = json.load(f)
    else:
        print("Running full ingestion pipeline (this may take several minutes)...")

        # Section-level split
        h1_pattern_pdf1 = re.compile(r'^(?!.*[_-])[1-9]\.\s+(?!.*\.{3,}\s*\d+$)[A-Za-z"&:;().,].*')
        h1_pattern_pdf2 = re.compile(r'^[1-9]\.0\s+(?!.*\.{3,})[A-Za-z].*')

        parallel_section = RunnableParallel(
            doc1=RunnableLambda(lambda x: splitting_section_pipeline(x["doc1"]["pdf_path"], x["doc1"]["h1_pattern"], 1)),
            doc2=RunnableLambda(lambda x: splitting_section_pipeline(x["doc2"]["pdf_path"], x["doc2"]["h1_pattern"], 2)),
        )
        section_result = parallel_section.invoke({
            "doc1": {"pdf_path": pdf_path1, "h1_pattern": h1_pattern_pdf1},
            "doc2": {"pdf_path": pdf_path2, "h1_pattern": h1_pattern_pdf2},
        })
        docs1_main_section = section_result["doc1"]
        docs2_main_section = section_result["doc2"]

        # Subsection-level split
        h2_pattern_pdf1 = re.compile(r"^[1-9][A-Z]?(\.\d+|[A-Z])?\.?\s+.*")
        h2_pattern_pdf2 = re.compile(r"^[1-9]\.\d\.?\s+.*")

        parallel_sub = RunnableParallel(
            doc1=RunnableLambda(lambda x: split_by_section(x["doc1"], h2_pattern_pdf1, "subsection")),
            doc2=RunnableLambda(lambda x: split_by_section(x["doc2"], h2_pattern_pdf2, "subsection")),
        )
        sub_result = parallel_sub.invoke({"doc1": docs1_main_section, "doc2": docs2_main_section})

        docs1 = reconstruct_docs(docs1_main_section, filter_subsections(sub_result["doc1"]))
        docs2 = reconstruct_docs(docs2_main_section, filter_subsections(sub_result["doc2"]))

        # Fine-grained chunking
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks1 = text_splitter.split_documents(docs1)
        chunks2 = text_splitter.split_documents(docs2)

        # Table extraction
        table_docs_1 = extract_table_docs(pdf_path1, chunks1, "PDF_1")
        table_docs_2 = extract_table_docs(pdf_path2, chunks2, "PDF_2")

        # Assign chunk IDs
        for chunk_id, chunk in enumerate(chunks1):
            chunk.metadata["chunk_id"] = chunk_id
        for chunk_id, chunk in enumerate(chunks2):
            chunk.metadata["chunk_id"] = chunk_id

        # Populate text vector stores
        chroma_vs_1.add_documents(chunks1)
        chroma_vs_1.add_documents(table_docs_1)
        chroma_vs_2.add_documents(chunks2)
        chroma_vs_2.add_documents(table_docs_2)

        # Populate image vector stores
        build_image_vector_store(docs1_main_section, chroma_img_1, "PDF_1")
        build_image_vector_store(docs2_main_section, chroma_img_2, "PDF_2")

        # Build and cache heading structure
        heading_structure = extract_heading_structure({
            "heading structure 1": {"metadatas": [d.metadata for d in docs1]},
            "heading structure 2": {"metadatas": [d.metadata for d in docs2]},
        })
        with open(heading_cache, "w", encoding="utf-8") as f:
            json.dump(heading_structure, f, ensure_ascii=False, indent=2)

        print("Ingestion complete.")

    state = AssistantState(heading_structure=heading_structure)
    return Assistant(model=model, state=state)
