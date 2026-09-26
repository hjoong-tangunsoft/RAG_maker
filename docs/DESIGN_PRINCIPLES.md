# 설계 원칙: Dumb Server, Smart Client

> 이 문서는 사내 RAG 자동 인제스트 파이프라인을 설계하며 확립한 원칙을 정리한다.
> 상세 풀 설명은 블로그 게시글 참고: [자동 인제스트 파이프라인 - Dumb Server, Smart Client](https://blog.tangunsoft.com/rag-dumb-server-smart-client)

---

## 한 줄 요약

> **RAG 서버는 dumb하게 유지하고, 모든 로직은 파이프라인 클라이언트에 몰빵한다.**

## 문제 - 자동 인제스트를 어떻게 설계할까

수동 인제스트로 몇 개 문서 넣고 테스트하는 단계는 끝났다. 실서비스는 사내 위키/문서 저장소와 RAG를 자동 동기화해야 한다:

- 사내 위키에 새 문서 올라오면 자동 반영
- 문서 수정되면 재인제스트
- 삭제된 문서는 알아서 사라짐
- Confluence, Notion, GHES Wiki, SharePoint 등 여러 소스 통합
- 폐쇄망(air-gapped) 환경 지원

이 요구사항을 만족하는 파이프라인 설계에는 반드시 결정해야 할 것이 있다: **책임을 어디에 둘 것인가?**

## 3가지 접근 비교

### 옵션 A: 스마트 라우팅 (분류기)

Apache 앞단에 분류기를 두고 "이 질문이 RAG 필요한지" 판단해서 지능적으로 라우팅.

- **문제 1**: 분류기 자체가 오버헤드. LLM 기반이면 매 요청마다 추가 호출.
- **문제 2**: 판단 실패 시 디버깅 지옥. "왜 얘는 문서 안 봤지?"
- **문제 3**: 사용자가 예측 못함.

**결론: 안 함.**

### 옵션 B: Smart Server (서버가 모든 로직)

`POST /rag/sync/confluence` 같은 엔드포인트에서 서버가 알아서 fetch → parse → chunk → embed → store.

- **문제 1**: 폐쇄망에서 외부 wiki 접근 불가 → 파이프라인 붕괴
- **문제 2**: 서버가 모든 wiki 인증 정보 알고 있어야 함
- **문제 3**: 위키 API 변경 시 서버 재배포
- **문제 4**: 새 문서 소스 추가마다 서버 코드 수정
- **문제 5**: 서버 로그가 유일한 진실 (관찰가능성 낮음)

**결론: 안 함.**

### 옵션 C: Dumb Server, Smart Client ⭐

RAG 서버는 "텍스트 받으면 청킹·임베딩·저장" 만 하는 dumb한 저장소로 유지. 소스별 특성 - 인증, 파싱, 정제 - 은 전부 클라이언트가 처리.

```
Wiki ──▶ [Pipeline Client]  ──HTTPS──▶ RAG 서버 (chunk + embed + store만)
           │
           ├─ fetch    (인증 여기서)
           ├─ parse    (PDF/HTML 정제 여기서)
           ├─ clean    (전처리 여기서)
           └─ POST /rag/ingest/text  ← 순수 텍스트만 전달
```

**얻는 것:**

- **새 문서 소스 추가** = 새 exporter 하나 추가. 서버 무손실
- **파싱 오류 발생** = 클라이언트에서 텍스트 확인 → 즉시 재시도
- **폐쇄망 배포** = 클라이언트가 폐쇄망 안에 있으면 됨
- **RAG 서버 다운타임** = 클라이언트 큐에 쌓아뒀다가 복구 후 flush
- **관찰가능성** = 파싱 결과가 파일로 남아 답변 품질 이슈 즉시 디버깅 가능

## 3가지 인제스트 API 자동화 관점

RAG 서비스는 3가지 API를 제공한다: `/rag/ingest/text`, `/rag/ingest/file`, `/rag/ingest/url`.

자동화 관점에서 비교:

| 관점 | /text | /file | /url |
|---|---|---|---|
| 재현성 | ✅ 같은 입력 → 같은 결과 | ⚠️ 서버 pypdf 버전 따라 다름 | ❌ 원격 페이지 바뀌면 결과 바뀜 |
| 디버깅 | ✅ JSON 로그로 정확한 입력 보존 | ⚠️ multipart 로그 복잡 | ❌ 어느 시점에 뭘 받았는지 불명 |
| 멱등성 | ✅ doc_id + 같은 텍스트 → 덮어쓰기 | ⚠️ 같은 파일 재업로드해도 재추출 | ❌ URL만 같음, 내용 다를 수 있음 |
| 폐쇄망 | ✅ 완벽 (외부 접속 불필요) | ✅ 로컬 파일이라 OK | ❌ 서버가 외부망 접속 시도 |
| 인증 문제 | ✅ 클라이언트가 처리 | ✅ 파일만 있으면 됨 | ❌ 서버가 wiki 인증 정보 알아야 함 |
| 전처리 유연성 | ✅ 클라에서 마음대로 정제 | ⚠️ 서버 로직 고정 | ❌ 서버가 크롤링 후 자동 처리 |

**결론: 자동화 파이프라인은 무조건 `/rag/ingest/text`.**

- `/file`은 사람이 브라우저로 한 번씩 파일 던지는 UI용
- `/url`은 실험·데모용

## 극단으로 밀면 - PDF도 클라이언트에서 처리

`/rag/ingest/file`을 안 쓰고 클라이언트에서 PDF → 텍스트 추출 후 `/rag/ingest/text`로 보낸다. 왜?

- **더 나은 파서 선택 가능**: `pdfplumber` (테이블), `nougat` (수식), `mistral-ocr` (스캔본) 등 상황별
- **재시도 로직 유연**: 1차 실패 → 2차 다른 파서 → 3차 OCR
- **파이프라인 로컬 캐시**: 파싱 결과 저장. 재실행 시 파싱 스킵
- **정제 로직**: 페이지 번호 제거, 목차 스킵, 반복 헤더 제거

파이프라인 흐름:

```
raw/handbook.pdf   ─┐
                   │ [client-side extract]
                   ▼
extracted/handbook.md   ─┐  (사람이 눈으로 확인 가능)
                        │ [state hash check]
                        ▼
POST /rag/ingest/text   → RAG 서버는 그냥 저장
```

중간 산출물 `extracted/handbook.md`가 파일로 남는 게 관건. 답변 품질이 이상하면 이 파일 열어서 실제로 뭐가 인덱싱됐는지 확인 가능. 서버 사이드 파싱은 이런 관찰가능성이 없다.

## 30줄 레퍼런스 파이프라인

원칙을 실제 코드로 옮기면 30줄이면 충분하다.

```python
import hashlib, json, httpx, pathlib

STATE_FILE = pathlib.Path("~/.rag-ingest/state.json").expanduser()
RAG_URL = "https://llm.tangunsoft.com/rag/ingest/text"

def sync_folder(folder: pathlib.Path):
    state = json.loads(STATE_FILE.read_text()) if STATE_FILE.exists() else {}
    for md_file in folder.glob("**/*.md"):
        doc_id = md_file.stem
        text = md_file.read_text(encoding="utf-8")
        content_hash = hashlib.sha256(text.encode()).hexdigest()
        if state.get(doc_id, {}).get("hash") == content_hash:
            print(f"  SKIP     {doc_id}")
            continue
        resp = httpx.post(RAG_URL, json={
            "doc_id": doc_id,
            "source": str(md_file.relative_to(folder)),
            "text": text,
            "metadata": {"path": str(md_file), "hash": content_hash},
        })
        resp.raise_for_status()
        state[doc_id] = {"hash": content_hash, "chunks": resp.json()["chunks"]}
        print(f"  INGEST   {doc_id} ({resp.json()['chunks']} chunks)")
    STATE_FILE.write_text(json.dumps(state, indent=2))

if __name__ == "__main__":
    sync_folder(pathlib.Path("./docs"))
```

핵심 3가지:

1. **콘텐츠 해시로 변경 감지** - 같은 텍스트는 재인제스트 안 함
2. **state 파일로 상태 추적** - 마지막 동기화 해시 기록. 증분 처리
3. **doc_id 명시** - 같은 doc_id 오면 서버가 자동으로 기존 청크 삭제 후 덮어쓰기

`crontab -e`나 `systemd timer`로 매일 새벽 자동 실행하면 사내 문서가 지속적으로 최신 상태 유지된다.

## 확장 - 4단계 로드맵

이 30줄을 실 운영 수준으로 키우는 로드맵. GitHub 이슈로 정리:

1. **[Export](https://github.com/hjoong-tangunsoft/RAG_maker/issues/11)** - 사내 위키/문서 소스에서 콘텐츠 뽑기 + 표준 중간 포맷 정의
2. **[Ingest](https://github.com/hjoong-tangunsoft/RAG_maker/issues/12)** - Export → Chroma 반영 + NEW/CHANGED/DELETED 증분 감지
3. **[CLI](https://github.com/hjoong-tangunsoft/RAG_maker/issues/13)** - `rag-ingest sync` 통합 명령 + systemd timer 자동 실행
4. **[폐쇄망](https://github.com/hjoong-tangunsoft/RAG_maker/issues/14)** - 인터넷 없이 배포 가능한 tarball 패키징

전체 로드맵: [Epic #16](https://github.com/hjoong-tangunsoft/RAG_maker/issues/16)

각 단계는 앞선 단계 위에 쌓이지만, **4단계(폐쇄망)는 크로스커팅 관심사**. 1~3단계 설계마다 "인터넷 없이도 되나?"를 체크해야 한다.

## 왜 스마트 라우팅은 안 하는가

Apache 앞단에 판단 로직을 두는 접근 - 이미 안 하기로 했다 (옵션 A). 이유는 사실 같은 원칙의 다른 얼굴이다.

RAG 원하는 클라이언트는 `/rag/v1`을 부르고, 순수 챗 원하는 클라이언트는 `/v1`을 부른다. Apache는 경로 접두어만 보고 라우팅한다. 판단은 클라이언트가 이미 URL 선택으로 끝냈다.

**판단은 dumb한 라우팅 계층 밖에서 하고, 라우터는 그저 문자열 매칭만 한다.** Dumb Server, Smart Client의 다른 얼굴.

## 정리

RAG 시스템의 확장성은 **서버의 지능**이 아니라 **책임 분리의 명확성**에서 온다.

- 서버는 "텍스트 받으면 저장" 만 함
- 클라이언트가 소스별 인증·파싱·정제·해시 감지·재시도 등 모든 로직 담당
- 파이프라인 프리컴퓨트 결과를 파일로 보존해서 관찰가능성 확보
- 폐쇄망은 클라이언트를 폐쇄망 안에 두면 자연스럽게 해결

이 원칙 하나만 지키면 새 소스 추가, 폐쇄망 대응, 디버깅, 재현성 문제가 자연스럽게 해결된다.

## 관련 문서

- [README.md](../README.md) - 운영 명령 · 튜닝 · 롤백
- [docs/ARCHITECTURE.md](ARCHITECTURE.md) - 시스템 아키텍처 설명
- [블로그: 사내 RAG 완전 구축기](https://blog.tangunsoft.com/rag-setup-fastapi-chroma-vllm) - Post 1
- [블로그: 자동 인제스트 파이프라인 - Dumb Server, Smart Client](https://blog.tangunsoft.com/rag-dumb-server-smart-client) - Post 2 (이 문서의 상세 버전)
