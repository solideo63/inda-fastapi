# ==================== main.py (FIXED - Query Expansion) ====================
# LangChain + Groq + Gemini + Qdrant Cloud with Smart Query Understanding

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager
import httpx
import os
import time
from dotenv import load_dotenv
import html as html_module
from datetime import datetime
import json
import re

# LangChain imports
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_qdrant import QdrantVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import Document

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

load_dotenv()

# API Keys
BPS_API_KEY = os.getenv("BPS_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Qdrant Cloud Configuration
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = "bps_statistics"





# -------------------- Initialize Components --------------------
print("\n" + "="*60)
print("📦 Initializing INDA Components...")
print("="*60)

# 1. Qdrant Cloud Client
print(f"🔗 Connecting to Qdrant Cloud...")
qdrant_client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
)

# 2. Embeddings Model
print("🧠 Loading embedding model...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'},
    cache_folder="/tmp/huggingface"
)

# 3. LLM Models
print("🤖 Initializing LLMs...")
llm_groq = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name="llama-3.1-8b-instant",
    temperature=0.3,
    max_tokens=2000
)

llm_gemini = ChatGoogleGenerativeAI(
    google_api_key=GOOGLE_API_KEY,
    model="gemini-2.0-flash-exp",
    temperature=0.3,
    max_tokens=2000
)

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq")
llm = llm_groq if LLM_PROVIDER == "groq" else llm_gemini

# 4. Create collection if not exists
print(f"📊 Checking collection '{COLLECTION_NAME}'...")
try:
    qdrant_client.get_collection(COLLECTION_NAME)
    print(f"✅ Collection '{COLLECTION_NAME}' exists")
except Exception:
    print(f"⚠️  Collection not found, creating...")
    qdrant_client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE)
    )
    print(f"✅ Collection '{COLLECTION_NAME}' created")

# 5. Vector Store
print("🗄️  Initializing vector store...")
vector_store = QdrantVectorStore(
    client=qdrant_client,
    collection_name=COLLECTION_NAME,
    embedding=embeddings
)

print("="*60)
print("✅ All components initialized successfully!")
print("="*60 + "\n")

# -------------------- Lifespan Event --------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    print("\n🚀 Starting INDA Backend...\n")
    
    try:
        collections = qdrant_client.get_collections()
        print(f"✅ Qdrant: {len(collections.collections)} collections")
        
        test_embed = embeddings.embed_query("test")
        print(f"✅ Embeddings: dim={len(test_embed)}")
        
        llm.invoke("test")
        print(f"✅ LLM: {LLM_PROVIDER} ready")
        
        count = qdrant_client.count(COLLECTION_NAME).count
        print(f"✅ Documents: {count} in vector store\n")
        
    except Exception as e:
        print(f"❌ Startup validation failed: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    yield
    
    print("\n👋 Shutting down INDA Backend...")

# -------------------- FastAPI App --------------------
app = FastAPI(
    title="INDA - Intelligent Data Assistant",
    version="2.0.0",
    lifespan=lifespan
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    
    print(f"\n{'='*60}")
    print(f"📥 {request.method} {request.url.path}")
    print(f"🌐 Origin: {request.headers.get('origin', 'N/A')}")
    
    response = await call_next(request)
    
    duration = time.time() - start_time
    print(f"📤 Status: {response.status_code}")
    print(f"⏱️  Duration: {duration:.2f}s")
    print(f"{'='*60}\n")
    
    return response

# -------------------- Models --------------------
class ChatRequest(BaseModel):
    question: str
    use_rag: bool = True
    model: str = "groq"

class ChatResponse(BaseModel):
    answer: str
    sources: list
    metadata: dict

# -------------------- Query Expansion Function --------------------
def expand_query_with_llm(question: str, selected_llm) -> dict:
    """Expand user query to extract keywords, time range, and context"""
    current_year = datetime.now().year
    
    expansion_prompt = f"""Kamu adalah asisten analisis query untuk sistem pencarian data BPS.

Pertanyaan user: "{question}"
Tahun saat ini: {current_year}

Tugasmu: Identifikasi kata kunci utama, rentang waktu, dan buat query yang diperluas.

PENTING: Output HANYA JSON murni tanpa teks tambahan apapun.

Format JSON:
{{"main_topic": "kata kunci utama", "keywords": ["kata kunci 1", "kata kunci 2"], "time_references": ["2024", "2025"], "expanded_query": "query yang diperluas dan jelas", "search_keyword": "kata kunci untuk API BPS"}}

Contoh:
Input: "inflasi 2 tahun belakangan"
Output: {{"main_topic": "inflasi", "keywords": ["inflasi", "IHK"], "time_references": ["{current_year-1}", "{current_year}"], "expanded_query": "data inflasi dan indeks harga konsumen tahun {current_year-1} dan {current_year}", "search_keyword": "inflasi"}}

Input: "data terbaru"
Output: {{"main_topic": "ekonomi", "keywords": ["ekonomi"], "time_references": ["{current_year}"], "expanded_query": "data statistik ekonomi terbaru tahun {current_year}", "search_keyword": "ekonomi"}}

Hanya output JSON:"""

    try:
        # Use lower temperature for consistent JSON
        if hasattr(selected_llm, 'temperature'):
            original_temp = selected_llm.temperature
            selected_llm.temperature = 0.1
        
        response = selected_llm.invoke(expansion_prompt)
        
        if hasattr(selected_llm, 'temperature'):
            selected_llm.temperature = original_temp
        
        result_text = response.content.strip()
        print(f"🤖 Raw LLM Response: {result_text[:200]}")
        
        # Extract JSON from response
        if "```json" in result_text:
            result_text = result_text.split("```json")[1].split("```")[0].strip()
        elif "```" in result_text:
            result_text = result_text.split("```")[1].split("```")[0].strip()
        
        # Try to find JSON object using regex
        json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', result_text, re.DOTALL)
        if json_match:
            result_text = json_match.group(0)
        
        expanded = json.loads(result_text)
        
        # Validate required fields
        if not all(key in expanded for key in ["main_topic", "expanded_query", "search_keyword"]):
            raise ValueError("Missing required fields in expansion result")
        
        print(f"✅ Query Expansion Success:")
        print(f"   Original: {question}")
        print(f"   Main Topic: {expanded.get('main_topic')}")
        print(f"   Expanded: {expanded.get('expanded_query')}")
        print(f"   Time: {expanded.get('time_references')}")
        print(f"   Search Keyword: {expanded.get('search_keyword')}")
        
        return expanded
        
    except Exception as e:
        print(f"⚠️  Query expansion failed: {str(e)}, using fallback")
        keyword = extract_keyword_from_question(question)
        return {
            "main_topic": keyword,
            "keywords": [keyword],
            "time_references": [str(current_year)],
            "expanded_query": question,
            "search_keyword": keyword
        }

def extract_keyword_from_question(question: str) -> str:
    """Extract relevant keyword from user question (fallback method)"""
    question_lower = question.lower()
    
    keyword_map = {
        "pengangguran": ["pengangguran", "tpt", "tingkat pengangguran terbuka", "angkatan kerja"],
        "inflasi": ["inflasi", "ihk", "indeks harga konsumen", "harga"],
        "kemiskinan": ["kemiskinan", "garis kemiskinan", "penduduk miskin"],
        "pdrb": ["pdrb", "pertumbuhan ekonomi", "ekonomi", "produk domestik"],
        "ipm": ["ipm", "indeks pembangunan manusia", "ipg"],
        "ekspor": ["ekspor", "impor", "neraca perdagangan"],
        "penduduk": ["penduduk", "demografi", "sensus"],
    }
    
    for main_keyword, variations in keyword_map.items():
        for variant in variations:
            if variant in question_lower:
                return main_keyword
    
    return "ekonomi"

# -------------------- Helper Functions --------------------
def clean_html(raw_html):
    """Clean HTML tags and decode HTML entities from text"""
    if not raw_html:
        return ""
    
    decoded = html_module.unescape(raw_html)
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', decoded)
    cleantext = re.sub(r'\s+', ' ', cleantext)
    
    return cleantext.strip()

async def fetch_bps_data(keyword: str, fetch_publications: bool = True):
    """Fetch data from BPS Web API"""
    DOMAIN = "1200"
    PAGE = 1
    LANG = "ind"
    url = "https://webapi.bps.go.id/v1/api/list"
    all_data = []
    
    # 1. Fetch Press Releases
    params_pr = {
        "model": "pressrelease",
        "lang": LANG,
        "domain": DOMAIN,
        "page": PAGE,
        "keyword": keyword,
        "key": BPS_API_KEY
    }
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(url, params=params_pr)
            resp.raise_for_status()
            data = resp.json()
            
            if data.get("status") == "OK" and "data" in data:
                data_content = data["data"]
                if isinstance(data_content, list) and len(data_content) > 1:
                    press_releases = data_content[1]
                    for item in press_releases:
                        if isinstance(item, dict):
                            item['source_type'] = 'press_release'
                            all_data.append(item)
                    print(f"✅ Fetched {len(press_releases)} press releases")
    except Exception as e:
        print(f"❌ Error fetching press releases: {e}")
    
    # 2. Fetch Publications
    if fetch_publications:
        params_pub = {
            "model": "publication",
            "lang": LANG,
            "domain": DOMAIN,
            "page": PAGE,
            "keyword": keyword,
            "key": BPS_API_KEY
        }
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.get(url, params=params_pub)
                resp.raise_for_status()
                data = resp.json()
                
                if data.get("status") == "OK" and "data" in data:
                    data_content = data["data"]
                    if isinstance(data_content, list) and len(data_content) > 1:
                        publications = data_content[1]
                        for item in publications:
                            if isinstance(item, dict):
                                item['source_type'] = 'publication'
                                all_data.append(item)
                        print(f"✅ Fetched {len(publications)} publications")
        except Exception as e:
            print(f"❌ Error fetching publications: {e}")
    
    print(f"📊 Total fetched: {len(all_data)} documents")
    return all_data

def save_to_qdrant_langchain(publications):
    """Save publications to Qdrant using LangChain"""
    documents = []
    
    for pub in publications:
        if not isinstance(pub, dict):
            continue
        
        title = clean_html(pub.get("title", "")).strip()
        abstract_raw = pub.get("abstract", "")
        abstract_full = clean_html(abstract_raw).strip()
        
        source_type = pub.get("source_type", "unknown")
        
        if not title:
            continue
        
        pub_id = str(pub.get("brs_id") if source_type == "press_release" 
                     else pub.get("pub_id", pub.get("id", "")))
        
        pdf_link = pub.get("pdf", "")
        if not pdf_link and source_type == "publication":
            pdf_link = pub.get("file_pdf", "")
        
        abstract_short = abstract_full[:250] + "..." if len(abstract_full) > 250 else abstract_full
        
        doc = Document(
            page_content=(
                f"Judul: {title}\n"
                f"Kategori: {pub.get('subcsa', pub.get('subj', ''))}\n"
                f"Abstrak: {abstract_full}\n"
            ),
            metadata={
                "pub_id": pub_id,
                "title": title,
                "abstract": abstract_short,
                "link": pdf_link,
                "release_date": pub.get("rl_date", pub.get("date", "")),
                "category": pub.get("subcsa", pub.get("subj", "")),
                "source_type": source_type,
            }
        )
        documents.append(doc)
    
    if documents:
        vector_store.add_documents(documents)
        print(f"✅ Saved {len(documents)} documents to Qdrant Cloud")

# -------------------- RAG Prompt (FIXED) --------------------
prompt_template = """
Kamu adalah <b>Ning Aida</b>, asisten virtual resmi dari <b>BPS Provinsi Sumatera Utara</b>. 
Tugasmu adalah menjawab pertanyaan pengguna secara <b>lengkap, akurat, dan berbasis data resmi</b> dari BPS. 
Gunakan <b>hanya informasi dari konteks dokumen</b> di bawah ini.

========================
📘 KONTEKS DOKUMEN:
{context}
========================
💬 PERTANYAAN:
{question}
========================
🧭 INSTRUKSI:
Tulis jawaban dalam format <b>HTML murni</b> (tanpa Markdown, tanpa angka urutan manual). 
Gunakan heading (<h2>, <h3>), paragraf (<p>), bullet list (<ul><li>), dan tabel (<table>) agar mudah dibaca.
Gunakan gaya penyampaian profesional, ramah, dan khas ASN BPS.

Struktur wajib jawaban:
<h2>Halo! Saya Ning Aida 👋</h2>
<p>Saya asisten data dari <b>BPS Provinsi Sumatera Utara</b> yang siap membantu Anda.</p>
<hr>

<h3>📊 Data Utama</h3>
<ul>
<li><b>[judul lengkap]</b> </li>
<li><b>Ringkasan Data:</b> [isi utama atau abstrak dari dokumen , utamakan dokumen berita resmi statistik]</li>
<li><b>Poin Penting:</b></li>
<ul>
<li>Gunakan 3–5 poin penting dari konteks terutama gunakan angka - angka data dari abstrak</li>
</ul>
<li><b>Link sumber:</b></li>
<ul>
<li><k>[pdf]</k></li>
</ul>
<hr>

<h3>📈 Tabel Indikator (Jika ada data numerik)</h3>
<table border="1" style="border-collapse:collapse;width:100%;text-align:center;">
<tr><th>Indikator</th><th>Nilai</th><th>Status</th><th>Tren</th></tr>
<tr><td>[nama indikator]</td><td>[angka]</td><td>⬆️/⬇️/➡️</td></tr>
</table>
<hr>

<h3>🔗 Sumber & Tautan</h3>
<ul>
<li><a href="[https://jatim.bps.go.id/id/publication?keyword=[keyword]&onlyTitle=false&sort=latest]" target="_blank">Baca Selengkapnya</a></li>
</ul>
<hr>

<h3>💡 Analisis & Insight</h3>
<ul>
<li><b>Fenomena Data:</b> Jelaskan tren atau perubahan penting.</li>
<li><b>Interpretasi:</b> Jelaskan makna dan dampaknya bagi masyarakat atau ekonomi.</li>
<li><b>Rekomendasi:</b> Saran kebijakan atau langkah tindak lanjut.</li>
</ul>
<hr>

<h3>📞 Layanan BPS</h3>
<p>BPS membuka pelayanan konsultasi data sebagai berikut:</p>
<ul>
<li>Senin–Kamis: 08.00–15.30</li>
<li>Jumat: 08.00–16.00</li>
</ul>
<p>Anda dapat berkonsultasi lebih lanjut melalui:
<a href="https://halopst.web.bps.go.id/konsultasi" target="_blank">Halo PST BPS</a></p>
<hr>

<h3>🙏 Penutup</h3>
<p>Semoga informasi ini membantu! Jangan ragu untuk bertanya lebih lanjut.</p>
<p>Silakan juga isi <a href="https://sumut.bps.go.id" target="_blank">Survei Kebutuhan Data</a> sebagai bentuk peningkatan layanan BPS Sumatera Utara.</p>

⚠️ PENTING:
- Jangan mengarang angka atau informasi yang tidak ada di konteks.
- Jika data tidak ditemukan, tulis dengan sopan:
  <i>"Maaf, data spesifik tentang [topik] belum tersedia pada dokumen saat ini. 
  Silakan kunjungi situs resmi <a href='https://sumut.bps.go.id' target='_blank'>BPS Sumut</a> untuk data terbaru."</i>

JAWABAN:
"""

# ✅ FIXED: Only use context and question as input variables
PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)

# ✅ DO NOT create global qa_chain - create per request instead

# -------------------- Endpoints --------------------
@app.get("/")
async def root():
    """Root endpoint"""
    try:
        count = qdrant_client.count(COLLECTION_NAME).count
        return {
            "app": "INDA - Intelligent Data Assistant",
            "version": "2.0.0 FIXED",
            "status": "healthy",
            "features": ["Query Expansion with LLM", "Smart Time Detection", "Multi-source RAG"],
            "framework": "LangChain",
            "llm_providers": ["Groq (Llama 3.1 8B)", "Google (Gemini 2.0 Flash)"],
            "vector_store": "Qdrant Cloud",
            "total_documents": count,
            "deployment": "HuggingFace Spaces",
        }
    except Exception as e:
        return {"app": "INDA", "status": "error", "error": str(e)}

@app.get("/api/health")
async def health_check():
    """Health check for all services"""
    status = {
        "fastapi": "healthy",
        "qdrant": "unknown",
        "llm_groq": "unknown",
        "llm_gemini": "unknown"
    }
    
    try:
        count = qdrant_client.count(COLLECTION_NAME).count
        status["qdrant"] = "healthy"
        status["qdrant_documents"] = count
    except Exception as e:
        status["qdrant"] = f"unhealthy: {str(e)}"
    
    try:
        llm_groq.invoke("test")
        status["llm_groq"] = "healthy"
    except Exception as e:
        status["llm_groq"] = f"unhealthy: {str(e)}"
    
    try:
        llm_gemini.invoke("test")
        status["llm_gemini"] = "healthy"
    except Exception as e:
        status["llm_gemini"] = f"unhealthy: {str(e)}"
    
    return status

@app.options("/api/chat")
async def chat_options():
    """Handle preflight OPTIONS request"""
    return {"status": "ok"}

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        question = request.question
        print(f"\n🔍 Original Question: {question}")
        
        # Select LLM
        global llm
        if request.model == "gemini":
            llm = llm_gemini
        else:
            llm = llm_groq
        
        # Expand query using LLM
        expanded_info = expand_query_with_llm(question, llm)
        expanded_query = expanded_info.get("expanded_query", question)
        search_keyword = expanded_info.get("search_keyword", "ekonomi")
        
        # Check document count
        count = qdrant_client.count(COLLECTION_NAME).count
        print(f"📊 Vector store has {count} documents")
        
        # Auto-fetch if needed
        if count < 10:
            print(f"⚠️ Vector store has few documents, fetching with keyword: {search_keyword}")
            publications = await fetch_bps_data(search_keyword, fetch_publications=True)
            if publications:
                save_to_qdrant_langchain(publications)
                print(f"✅ Added {len(publications)} documents")
        
        # ✅ Create fresh chain for each request
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vector_store.as_retriever(
                search_kwargs={
                    "k": 3,
                    "score_threshold": 0.5
                }
            ),
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT}
        )
        
        # ✅ Run RAG with expanded query
        result = qa_chain.invoke({"query": expanded_query})
        
        # Add expansion info to answer
        expansion_info_html = f"""<div style="background-color: #EFF6FF; border-left: 4px solid #3B82F6; padding: 12px; margin-bottom: 16px; border-radius: 4px;">
<p style="margin: 0; font-size: 14px;"><strong>🧠 Pemahaman Query AI:</strong></p>
<p style="margin: 4px 0 0 0; font-size: 13px;">
<strong>Pertanyaan asli:</strong> {question}<br>
<strong>Diperluas menjadi:</strong> <em>{expanded_query}</em><br>
<strong>Kata kunci pencarian:</strong> {search_keyword}
{"<br><strong>Periode waktu:</strong> " + ", ".join(expanded_info.get("time_references", [])) if expanded_info.get("time_references") else ""}
</p>
</div>

"""
        
        enhanced_answer = expansion_info_html + result["result"]
        
        # Extract sources
        sources = []
        for doc in result.get("source_documents", []):
            metadata = doc.metadata
            sources.append({
                "title": metadata.get("title"),
                "abstract": metadata.get("abstract", ""),
                "link": metadata.get("link"),
                "release_date": metadata.get("release_date"),
                "source_type": "📰 Press Release" if metadata.get("source_type") == "press_release" else "📚 Publikasi",
                "category": metadata.get("category"),
                "similarity": 0
            })
        
        return ChatResponse(
            answer=enhanced_answer,
            sources=sources,
            metadata={
                "original_query": question,
                "expanded_query": expanded_query,
                "search_keyword": search_keyword,
                "time_references": expanded_info.get("time_references", []),
                "found_documents": len(sources),
                "model": request.model,
            }
        )
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/clear")
async def clear_collection():
    """Clear all documents and recreate collection"""
    try:
        qdrant_client.delete_collection(COLLECTION_NAME)
        qdrant_client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE)
        )
        return {"success": True, "message": "Collection cleared"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/refresh")
@app.post("/api/refresh")
async def refresh_data(keyword: str = "inflasi"):
    """Refresh vector store with latest BPS data"""
    try:
        print(f"\n🔄 Refreshing data with keyword: {keyword}")
        
        publications = await fetch_bps_data(keyword, fetch_publications=True)
        
        if publications:
            save_to_qdrant_langchain(publications)
            print(f"✅ Successfully fetched {len(publications)} documents")
        else:
            print("⚠️ No documents found")
        
        count = qdrant_client.count(COLLECTION_NAME).count
        
        pr_count = sum(1 for p in publications if p.get('source_type') == 'press_release')
        pub_count = sum(1 for p in publications if p.get('source_type') == 'publication')
        
        return {
            "success": True,
            "message": f"Refreshed with keyword '{keyword}'",
            "fetched_total": len(publications),
            "fetched_press_releases": pr_count,
            "fetched_publications": pub_count,
            "total_documents_in_db": count,
            "keyword": keyword
        }
    except Exception as e:
        print(f"❌ Refresh error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))