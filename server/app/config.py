"""Runtime configuration loaded from environment variables."""
from __future__ import annotations

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file="/upload/rag/rag.env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Filesystem layout
    data_dir: Path = Path("/upload/rag/data")
    chroma_dir: Path = Path("/upload/rag/data/chroma")
    docs_dir: Path = Path("/upload/rag/data/docs")
    models_dir: Path = Path("/upload/rag/models")

    # Server bind
    host: str = "127.0.0.1"
    port: int = 8100

    # Auth: if set, every request except /health must send X-API-Key
    api_key: str | None = None

    # Downstream LLM (LiteLLM proxy)
    litellm_url: str = "http://127.0.0.1:4000"
    litellm_api_key: str = "sk-1234"  # matches LITELLM_MASTER_KEY
    default_model: str = "qwen2.5-7b"

    # Embedding model (CPU)
    embed_model_name: str = "intfloat/multilingual-e5-base"
    embed_batch_size: int = 32
    # E5 family requires prefixes for asymmetric passage/query encoding
    embed_query_prefix: str = "query: "
    embed_passage_prefix: str = "passage: "

    # Chunking
    chunk_size: int = 800  # chars
    chunk_overlap: int = 120  # chars

    # Retrieval
    top_k: int = 5
    # Anti-Hallucination Guard L2: filter out weak/tangential retrieval hits.
    # E5 multilingual embeddings on Korean queries score related docs 0.7-0.9
    # and unrelated docs 0.3-0.5. 0.55 sits in the safe middle band, keeping
    # legitimate hits while blocking noise that enables bridging hallucination
    # (e.g. "승민소프트" query matching 탄군소프트 docs at 0.79 gets kept, but
    # completely unrelated queries lose their weakest hits).
    min_score: float = 0.55

    # Generation
    # Temperature lowered 0.2 -> 0.1 (Korean Purity Guard L2).
    # Qwen 2.5 leaks 한자 less at lower temps. Override via API for creative tasks.
    rag_temperature: float = 0.1
    rag_max_tokens: int = 1024
    # Aggressive Korean-only system prompt (Korean Purity Guard L1).
    # Qwen 2.5 heavily trained on Chinese; explicit prohibition + variant
    # examples keep answers in pure Hangul when the question is Korean.
    rag_system_prompt: str = (
        "당신은 정확한 한국어 어시스턴트입니다. 반드시 아래 규칙을 따르세요:\n\n"
        "1. 제공된 컨텍스트만 사용해서 답변하세요. 컨텍스트에 없는 정보는 지어내지 마세요.\n"
        "2. 컨텍스트가 부족하거나 관련 없으면 솔직히 '자료에 없습니다'라고 답하세요.\n"
        "3. 사용한 컨텍스트 번호를 [n] 형식으로 인용하세요.\n"
        "4. **한국어 질문에는 반드시 순수 한국어(한글)로만 답변하세요.**\n"
        "5. **한자(漢字, 중국어 문자) 사용 금지.** 한자어는 한글로 표기하세요:\n"
        "   예: 業務->업무, 會社->회사, 資料->자료, 情報->정보, 顧客->고객, 提供->제공\n"
        "6. 사용자가 다른 언어(영어/중국어 등)로 물으면 그 언어로 답변하세요.\n"
        "   단, 한국어 질문에 중국어를 섞는 것은 절대 금지입니다.\n"
        "7. 프로그래밍 코드나 명령어는 원문 그대로 유지하세요.\n"
        "8. **URL 은 마크다운 링크 `[텍스트](url)` 형식으로 표시하지 마세요.**\n"
        "   반드시 백틱으로 감싼 순수 URL 로 표시하세요:\n"
        "   - 잘못된 예: 자세한 내용은 [이 링크](https://example.com/foo)를 참조하세요.\n"
        "   - 올바른 예: 자세한 내용은 `https://example.com/foo` 를 참조하세요.\n"
        "   이유: 사용자가 URL 을 클릭하면 클라이언트 내장 브라우저가 열려\n"
        "   로그인 세션이 공유되지 않아 흰 화면이 뜨는 문제를 회피하기 위함입니다.\n"
        "   백틱으로 감싼 URL 은 클릭 불가 코드 텍스트로 표시되어 사용자가\n"
        "   복사·붙여넣기 로 로그인된 브라우저에서 열 수 있습니다."
    )
    # Post-hoc guard (Korean Purity Guard L3): if the LLM response contains
    # this many CJK Unified Ideographs (한자/漢字), regenerate with a
    # reinforced prompt. Phase E lowered from 2 → 0 (zero tolerance).
    # Hangul (한글, 0xAC00-0xD7AF) is a separate unicode range and never
    # triggers this threshold.
    hanja_threshold: int = 0

    # Citations display (Issue #9 Option A)
    # When True, /rag/query and /rag/v1/chat/completions (rag=true) append a
    # markdown footer listing sources to the answer body, so clients that
    # don't parse the extra `citations` JSON field (e.g. Continue.dev) still
    # see sources rendered as text. Disable to keep pure LLM output.
    append_citations_to_body: bool = True


settings = Settings()

# Ensure runtime directories exist
for p in (settings.data_dir, settings.chroma_dir, settings.docs_dir, settings.models_dir):
    p.mkdir(parents=True, exist_ok=True)
