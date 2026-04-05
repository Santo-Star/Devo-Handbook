import streamlit as st
import glob
import base64
import os
import time
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

# Configuración de Streamlit
st.set_page_config(
    page_title="Devocion AI | Manual del Empleado",
    layout="wide",
    page_icon="☕",
    initial_sidebar_state="expanded"
)

# --- UTILIDADES ---
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

# --- SISTEMA DE DISEÑO PREMIUM ---
def load_css():
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
        
        :root {
            --primary: #D4AF37;
            --primary-glow: rgba(212, 175, 55, 0.4);
            --bg-dark: #050505;
            --card-dark: #121212;
            --sidebar-dark: #0a0a0a;
            --text-main: #f8fafc;
            --text-muted: #8e9196;
            --accent-coffee: #C29B61;
        }

        /* General & Scrollbar Customization */
        * {
            font-family: 'Outfit', sans-serif !important;
            line-height: 1.6;
        }
        
        ::-webkit-scrollbar {
            width: 6px;
        }
        ::-webkit-scrollbar-track {
            background: #050505;
        }
        ::-webkit-scrollbar-thumb {
            background: #333;
            border-radius: 10px;
        }
        ::-webkit-scrollbar-thumb:hover {
            background: var(--primary);
        }

        .stApp {
            background: radial-gradient(circle at 50% -20%, #1e1e1e, #050505);
            color: var(--text-main);
        }

        /* Sidebar Glassmorphism Extra */
        section[data-testid="stSidebar"] {
            background-color: var(--sidebar-dark) !important;
            border-right: 1px solid rgba(212, 175, 55, 0.1);
        }
        
        section[data-testid="stSidebar"] h1 {
            font-size: 1.3rem !important;
            letter-spacing: 3px;
            background: linear-gradient(90deg, var(--primary), #fff);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-transform: uppercase;
        }

        /* Primary UI Buttons */
        .stButton>button {
            width: 100%;
            border-radius: 12px;
            background: linear-gradient(135deg, var(--primary) 0%, #b8860b 100%);
            color: black !important;
            border: none;
            padding: 0.7rem;
            font-weight: 700;
            transition: all 0.3s ease;
            box-shadow: 0 4px 20px rgba(212, 175, 55, 0.2);
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        
        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 25px rgba(212, 175, 55, 0.4);
            color: black !important;
        }

        /* Header Premium Dramatic */
        .main .block-container {
            padding-top: 0rem !important;
            margin-top: -8.5rem !important;
        }
        
        .main-header {
            background: linear-gradient(to right, #FFD700, #FFF, #DAA520);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 800;
            font-size: 4rem;
            margin-bottom: -15px;
            letter-spacing: -2px;
        }
        
        .sub-header {
            color: var(--text-muted) !important;
            font-size: 1.15rem;
            margin-top: -20px !important;
            margin-bottom: 1.5rem !important;
            font-weight: 300;
        }

        /* Chat UI Evolution */
        .visor-header, .chat-header-wrapper, 
        .visor-header *, .chat-header-wrapper * {
            font-size: 1.8rem;
            font-weight: 700;
            color: #FFD700 !important;
            letter-spacing: -0.5px;
            margin-top: 0px !important;
            padding-top: 0px !important;
            white-space: nowrap;
        }

        .chat-header-wrapper {
            display: flex;
            align-items: center;
            gap: 12px;
        }

        div[data-testid="stChatMessage"] {
            border-radius: 24px !important;
            padding: 24px !important;
            margin-bottom: 20px !important;
            border: 1px solid rgba(255,255,255,0.03) !important;
            transition: all 0.3s ease;
        }
        
        /* Assistant Bubble */
        div[data-testid="stChatMessage"][data-testid="assistant"] {
            background: rgba(18, 18, 18, 0.8) !important;
            border-left: 2px solid var(--primary) !important;
            box-shadow: -10px 0 30px rgba(212, 175, 55, 0.05);
        }
        
        /* User Bubble */
        div[data-testid="stChatMessage"][data-testid="user"] {
            background: rgba(255, 255, 255, 0.03) !important;
            border-right: 2px solid #555 !important;
        }

        /* Typography spacing for reading */
        div[data-testid="stChatMessageContent"] {
            font-size: 1.1rem !important;
            letter-spacing: 0.2px;
            color: #e2e8f0 !important;
        }

        /* PDF Viewer Container Glow */
        .pdf-container {
            background: #000;
            padding: 2px;
            border-radius: 20px;
            box-shadow: 0 30px 60px rgba(0,0,0,0.8);
            border: 1px solid rgba(212, 175, 55, 0.15);
            overflow: hidden;
        }

        h1, h2, h3, h4, h5, h6, p, span {
            color: var(--text-main) !important;
        }
        
        /* Input Field Styling */
        div[data-testid="stChatInput"] {
            border-top: 1px solid rgba(212, 175, 55, 0.1);
            background: transparent !important;
        }
        
        /* Hide default Streamlit marks */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        
        [data-testid="stChatMessageAvatarUser"], 
        [data-testid="stChatMessageAvatarAssistant"] {
            display: none !important;
        }
    </style>
    """, unsafe_allow_html=True)

load_css()

def add_floating_logo(img_name="Devo3.gif"):
    # Estrategia de Incrustación Base64 definitiva para la Nube
    # Esto elimina cualquier problema de rutas (/static/ vs /app/static/)
    try:
        # Buscamos la imagen en la carpeta static local
        img_path = os.path.join("static", img_name)
        if os.path.exists(img_path):
            base64_img = get_base64_of_bin_file(img_path)
            prefix = "image/gif" if img_name.endswith(".gif") else "image/avif"
        else:
            return # No mostramos nada si no existe el archivo
            
        st.markdown(
            f"""
                <style>
                .chat-companion {{
                    width: 400px;
                    display: block;
                    margin-left: auto;
                    margin-top: -120px !important;
                    filter: drop-shadow(0 0 35px rgba(194, 155, 97, 0.5));
                    animation: float 4s ease-in-out infinite;
                    z-index: 10;
                    image-rendering: auto; 
                }}
                @keyframes float {{
                    0% {{ transform: translateY(0px) rotate(0deg); }}
                    50% {{ transform: translateY(-15px) rotate(2deg); }}
                    100% {{ transform: translateY(0px) rotate(0deg); }}
                }}
                @media (max-width: 768px) {{
                    .chat-companion {{ width: 200px; margin: 0 auto; }}
                }}
                </style>
                <img src="data:{prefix};base64,{base64_img}" class="chat-companion">
                """,
                unsafe_allow_html=True
            )
    except Exception:
        pass # Silenciamos errores de carga para no romper la UI

# --- LÓGICA DE PROCESAMIENTO ---

@st.cache_resource
def get_pdf_path():
    pdf_files = glob.glob("*.pdf")
    if pdf_files:
        return pdf_files[0]
    return None

def secure_batch_indexing(texts, embeddings, batch_size=1):
    vector_store = None
    progress_bar = st.progress(0, text="Sincronizando manual con IA (Fase de Cuota)...")
    
    total_batches = (len(texts) + batch_size - 1) // batch_size
    
    for i in range(0, len(texts), batch_size):
        current_batch = i // batch_size + 1
        batch_texts = texts[i : i + batch_size]
        
        if vector_store is None:
            vector_store = FAISS.from_texts(batch_texts, embedding=embeddings)
        else:
            vector_store.add_texts(batch_texts)
        
        progress_val = int((current_batch / total_batches) * 100)
        progress_bar.progress(progress_val, text=f"Digeriendo el manual... ({current_batch}/{total_batches})")
        
        if current_batch < total_batches:
            time.sleep(5)
            
    progress_bar.empty()
    return vector_store

@st.cache_resource(show_spinner=False)
def initialize_rag(pdf_path, raw_api_key):
    if not pdf_path or not raw_api_key:
        return None, None
    
    # Usamos la API Key desde el entorno o st.secrets (para la nube)
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key and "GOOGLE_API_KEY" in st.secrets:
        api_key = st.secrets["GOOGLE_API_KEY"]
        
    if not api_key:
        st.error("❌ No se encontró GOOGLE_API_KEY en el entorno ni en secrets.")
        return None, None
        
    os.environ["GOOGLE_API_KEY"] = api_key
    genai.configure(api_key=api_key)
        
    try:
        reader = PdfReader(pdf_path)
        content = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                content.append(text)
        
        all_text = "\n\n".join(content)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
        chunks = text_splitter.split_text(all_text)
        
        # --- AUTO-DESCUBRIMIENTO DINÁMICO DE MODELOS ---
        models = list(genai.list_models())
        
        # 1. Buscar Modelo de Embeddings
        embed_models = [m.name for m in models if 'embedContent' in m.supported_generation_methods]
        if not embed_models:
            raise Exception("No se encontraron modelos de embeddings disponibles.")
        best_embed_model = embed_models[0]
        
        # 2. Buscar Modelo de Chat (LLM)
        chat_models = [m.name for m in models if 'generateContent' in m.supported_generation_methods]
        if not chat_models:
            raise Exception("No se encontraron modelos de chat disponibles.")

        # Filtramos modelos válidos de chat (que no sean puramente de visión)
        valid_chat_models = [m for m in chat_models if "vision" not in m.lower()]
        
        # PRIORIZACIÓN DE ESTABILIDAD BASADA EN DISPONIBILIDAD REAL:
        # Buscamos en orden de preferencia según lo que Google suele ofrecer
        preferred_model = None
        priority_keywords = ["gemini-1.5-flash-latest", "gemini-1.5-flash", "gemini-flash-latest", "gemini-2.0-flash", "gemini-pro"]
        
        for kw in priority_keywords:
            preferred_model = next((m for m in valid_chat_models if kw in m.lower()), None)
            if preferred_model:
                break
        
        # Fallback final: Lo primero que encontremos si no hay ninguno de los de arriba
        if not preferred_model:
            preferred_model = valid_chat_models[0]
        
        st.session_state.available_models = valid_chat_models
        st.session_state.default_model = preferred_model
        
        # Si el usuario ya seleccionó uno en la sesión, lo respetamos
        current_model = st.session_state.get("selected_model", preferred_model)
        
        embeddings = GoogleGenerativeAIEmbeddings(model=best_embed_model, google_api_key=api_key)
        vector_store = secure_batch_indexing(chunks, embeddings, batch_size=1)
        
        retriever = vector_store.as_retriever(search_kwargs={"k": 5})
        llm = ChatGoogleGenerativeAI(model=current_model, google_api_key=api_key, temperature=0.1)
        
        system_prompt = (
            "Eres el Asistente Experto de 'Café Devoción'. "
            "Tu misión es responder preguntas basadas en el manual proporcionado.\n\n"
            "CONTEXTO DEL MANUAL:\n{context}\n\n"
            "-------------------\n"
            "REGLA DE ORO DE COMUNICACIÓN (OBLIGATORIA):\n"
            "1. Identifica el idioma de la pregunta del usuario.\n"
            "2. Responde EXCLUSIVAMENTE en ese mismo idioma (si preguntan en Inglés, responde en Inglés; si es en Español, responde en Español).\n"
            "3. Asegura precisión, fluidez y consistencia en la comunicación.\n"
            "4. No cambies de idioma a menos que el usuario lo solicite explícitamente.\n"
            "5. Si la información no está en el manual, informa en el mismo idioma del usuario que no dispones de esos datos.\n"
            "-------------------"
        )
        
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])
        
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        rag_chain = (
            {
                "context": itemgetter("input") | retriever | format_docs, 
                "input": itemgetter("input")
            }
            | prompt_template
            | llm
            | StrOutputParser()
        )
        
        return rag_chain, None
    except Exception as e:
        st.error(f"Error al inicializar el sistema RAG: {e}")
        return None, None

import streamlit.components.v1 as components

def display_pdf_viewer(file_path):
    # Intentamos cargar el visor nativo pero optimizado
    pdf_name = os.path.basename(file_path)
    
    # RUTA OFICIAL DE STREAMLIT PARA ARCHIVOS ESTATICOS
    # Nota: Si el visor sigue en blanco, es bloqueo de Chrome/Edge
    pdf_url = f"/static/{pdf_name}"

    st.info("💡 **Consejo:** Si el visor de abajo no aparece o dice 'Bloqueado', haz clic en el botón de la derecha para abrir el manual en una pestaña nueva.")
    
    col_v, col_d = st.columns([4, 1])
    
    with col_v:
        # Usamos OBJECT en vez de IFRAME (A veces Chrome lo prefiere)
        st.markdown(f"""
            <div class="pdf-container" style="background: black; padding: 2px; border-radius: 12px; border: 1px solid #D4AF37;">
                <object data="{pdf_url}" type="application/pdf" width="100%" height="850px">
                    <embed src="{pdf_url}" type="application/pdf" />
                </object>
            </div>
        """, unsafe_allow_html=True)
        
    with col_d:
        with open(file_path, "rb") as f:
            st.download_button("📥 Descargar", f, f"{pdf_name}", "application/pdf")
        
        # LINK DIRECTO: Esta es la clave. Si el visor falla, este botón lo abre en pestaña nueva sin errores.
        st.link_button("📖 Ver Pantalla Completa", pdf_url)


# --- APLICACIÓN PRINCIPAL ---

def main():
    with st.sidebar:
        # Usando el nuevo logo oficial local (Ruta directa de disco)
        st.image("static/logo.avif", width=220)
        st.title("Admin Console")
        
        # La API Key ahora se carga automáticamente (Entorno o Secrets)
        env_key = os.getenv("GOOGLE_API_KEY") or st.secrets.get("GOOGLE_API_KEY")
        if env_key:
            st.success("🔐 **Conexión Google AI Activa**")
        else:
            st.warning("⚠️ **GOOGLE_API_KEY no detectada**. \n\nAsegúrate de configurarla en Secrets de la nube.")
        
        pdf_path = get_pdf_path()
        if pdf_path:
            st.success(f"📄 **Manual Detectado**\n\n{os.path.basename(pdf_path)}")
        else:
            st.error("❌ No se encontró el manual (.pdf)")

        st.divider()
        st.subheader("🛠️ Administración")
        
        # Botón para activar sincronización
        if st.button("🔄 Sincronizar Manual"):
            st.session_state.sync_requested = True
            st.session_state.is_syncing = True # Activamos estado visual de sincronización
            st.rerun()
            
        if st.button("🧹 Limpiar Chat"):
            st.session_state.messages = []
            st.rerun()
        
        st.subheader("⚙️ Configuración")
        
        if "available_models" in st.session_state:
            # Limpiar nombres de modelos para el selector
            model_options = st.session_state.available_models
            try:
                current_idx = model_options.index(st.session_state.get("selected_model", st.session_state.default_model))
            except ValueError:
                current_idx = 0
                
            selected = st.selectbox(
                "Modelo de IA",
                options=model_options,
                index=current_idx,
                help="Si recibes errores 429 (Quota Exceeded), intenta cambiar a gemini-1.5-flash."
            )
            
            if selected != st.session_state.get("selected_model"):
                st.session_state.selected_model = selected
                st.info("Modelo actualizado. Sincronizando...")
                st.rerun()

        st.caption("Tip: Gemini 1.5 Flash tiene cuotas más altas en el modo gratuito.")

    st.markdown('<h1 class="main-header">Asistente Virtual Devoción</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Tu guía inteligente para el Manual del Empleado. Consulta políticas, beneficios y normativas en segundos.</p>', unsafe_allow_html=True)

    # El interruptor de vista ahora se encuentra debajo de la descripción principal
    tcol1, tcol2 = st.columns([1, 4])
    with tcol1:
        show_viewer = st.toggle("📖 Ver Manual (Visor)", value=False)

    # --- MOSTRAR MASCOTA Y DISEÑO INICIAL (Presente desde el inicio) ---
    if show_viewer:
        col1, col2 = st.columns([1, 1.2], gap="large")
    else:
        # Chat centrado si no hay visor
        _padL, col2, _padR = st.columns([0.5, 3, 0.5], gap="medium")
        col1 = None # No hay columna para el PDF
    
    with col2:
        title_col, companion_col = st.columns([1.5, 1], gap="small")
        with title_col:
            st.markdown('<div class="chat-header-wrapper"><span>💬</span><span>Chat con el Manual</span></div>', unsafe_allow_html=True)
        with companion_col:
            # Determinamos qué mascota mostrar: Devo4.gif durante sync, Devo3.gif después
            mascot = "Devo4.gif" if st.session_state.get("is_syncing", False) else "Devo3.gif"
            add_floating_logo(mascot) 
            
    if col1:
        with col1:
            st.markdown('<div class="visor-header">📖 Visor del Documento</div>', unsafe_allow_html=True)
            if pdf_path:
                display_pdf_viewer(pdf_path)

    # --- INICIALIZACIÓN RAG (SOLO SI SE PRESIONA EL BOTÓN) ---
    if not st.session_state.get("sync_requested", False):
        st.info("👈 Presiona el botón 'Sincronizar Manual' en el Admin Console para comenzar.")
        return

    api_key_env = os.getenv("GOOGLE_API_KEY") or st.secrets.get("GOOGLE_API_KEY")
    if not api_key_env:
        st.warning("👈 Por favor, configura la GOOGLE_API_KEY en secrets.")
        return

    with st.spinner("Sincronizando con el Manual del Empleado... ☕"):
        # null passed since we use it directly inside initialize_rag now
        rag_chain, _ = initialize_rag(pdf_path, "from_env")
        # Una vez termina la sincronización, desactivamos el estado visual
        if st.session_state.get("is_syncing", False):
            st.session_state.is_syncing = False
            st.rerun()

    if not rag_chain:
        return

    # --- CONTINUACIÓN DEL CHAT ---
    with col2:
        if "messages" not in st.session_state:
            st.session_state.messages = []

        chat_container = st.container(height=750)
        with chat_container:
            for message in st.session_state.messages:
                # Ocultados via CSS - llamadas limpias
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

        if user_prompt := st.chat_input("Escribe tu pregunta aquí..."):
            with chat_container:
                with st.chat_message("user"):
                    st.markdown(user_prompt)
            st.session_state.messages.append({"role": "user", "content": user_prompt})

            with chat_container:
                with st.chat_message("assistant"):
                    with st.spinner("Revisando el manual..."):
                        try:
                            # Implementación de reintentos simples para errores 429
                            max_retries = 3
                            for attempt in range(max_retries):
                                try:
                                    answer = rag_chain.invoke({"input": user_prompt})
                                    st.markdown(answer)
                                    st.session_state.messages.append({"role": "assistant", "content": answer})
                                    break
                                except Exception as e:
                                    if "429" in str(e) and attempt < max_retries - 1:
                                        wait_time = (attempt + 1) * 5
                                        st.warning(f"Límite de cuota alcanzado. Reintentando en {wait_time}s... (Intento {attempt + 1}/{max_retries})")
                                        time.sleep(wait_time)
                                        continue
                                    else:
                                        raise e
                        except Exception as e:
                            error_msg = str(e)
                            if "429" in error_msg:
                                st.error("🚫 **Cuota Agotada (Error 429)**: Has excedido el límite de tu API Key de Google. \n\n**Soluciones:**\n1. Espera unos minutos antes de preguntar de nuevo.\n2. Prueba cambiando a 'gemini-1.5-flash' en el menú lateral.\n3. Asegúrate de no tener otros procesos usando la misma clave.")
                            else:
                                st.error(f"Error procesando la respuesta: {e}")


if __name__ == "__main__":
    main()
