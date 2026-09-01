# scripts/ — 원시 vs CLI 두 버전

이 폴더에는 RAG 인제스트 파이프라인이 **두 가지 형태**로 나란히 존재한다. 둘 다 **똑같은 로직 · 똑같은 결과**를 낸다. 차이는 **사용성·구조·확장성** 뿐이다.

> 왜 두 개 다 유지? 학습·비교용. 원시 스크립트는 "30줄 원칙"의 최소 형태를 보여주고, CLI 패키지는 실 운영 수준으로 확장한 결과를 보여준다.

## 파일 지도

```
scripts/
├── rag_sync.py                 ★ 원시 - 85 LOC, 단일 파일, systemd로 직접 실행
├── rag-ingest                  ★ CLI - shell wrapper (/usr/local/bin/ 배포용)
├── rag_ingest/                 ★ CLI 패키지 - 모듈 분리
│   ├── __init__.py               버전 정의
│   ├── __main__.py               python -m rag_ingest 진입점
│   ├── cli.py                    ★ argparse 껍데기 (subcommand 정의)
│   ├── core.py                   sync() 핵심 로직 (rag_sync.py에서 refactor)
│   ├── state.py                  load/save state + sha256
│   └── ingest_client.py          HTTP 클라이언트 (POST/GET/DELETE)
├── smoke_test.sh                 (기존) 배포 검증
├── eval_embedding.py             (기존) 임베딩 품질 테스트
└── README.md                     이 파일
```

## 원시 스크립트 (`rag_sync.py`)

**특징:**
- 단일 파일 85 LOC
- Config 는 오직 환경 변수 (`POOL_DIR`, `STATE_FILE`, `RAG_URL`, `DAILY_LIMIT`)
- 항상 `sync()` 하나만 수행하고 종료
- systemd timer 대상으로 설계

**호출:**
```bash
POOL_DIR=/upload/rag/data/docs-pool \
STATE_FILE=/upload/rag/data/sync-state.json \
RAG_URL=http://127.0.0.1:8100/rag/ingest/text \
DAILY_LIMIT=2 \
  /upload/rag/venv/bin/python3 /upload/rag/scripts/rag_sync.py
```

**언제 쓸까?**
- 배치 잡 (systemd/cron) 만 돌리는 상황
- 서버가 dumb한 만큼 클라이언트도 dumb하게 유지하고 싶을 때
- "30줄 원칙"의 최소 형태를 보고 싶을 때

## CLI 패키지 (`rag_ingest/`)

**특징:**
- 6개 서브커맨드: `sync`, `status`, `ls`, `refresh`, `reset`, `health`
- `--help`, `--dry-run`, `--limit`, `--only`, `--now`, `--confirm` 등 옵션 다수
- 재사용 가능한 `sync()` 함수 · `IngestClient` 클래스로 분리
- 사용자 시나리오 (관리자가 손으로 조작) 에 최적화

**호출:**
```bash
rag-ingest --help
rag-ingest sync                    # 매일 자동 실행되는 것과 동일
rag-ingest status                  # 로컬 뷰 (pool + state 파일)
rag-ingest ls                      # 서버 뷰 (/rag/docs 호출)
rag-ingest sync --dry-run          # 어떤 게 될지 미리 보기
rag-ingest sync --limit 10         # 오늘 10개까지 인제스트
rag-ingest sync --only foo.md      # 특정 파일만
rag-ingest refresh foo.md --now    # 강제 재인제스트
rag-ingest reset --confirm         # state 초기화
rag-ingest health                  # 서버 상태
```

**언제 쓸까?**
- 관리자가 손으로 조작할 일이 있을 때 (문서 재인제스트, 상태 조회 등)
- 여러 팀원이 파이프라인을 만질 때 (학습 곡선 낮춤)
- 여러 소스 통합 등 조합이 복잡해질 때 (`--only`, `--limit` 조합)
- 신입 온보딩 시 `rag-ingest --help` 하나로 자기 파악하게 하고 싶을 때

## 로직은 실제로 동일함 (증명)

원시 vs CLI 의 sync 로직 대응:

| 원시 (`rag_sync.py`) | CLI (`rag_ingest/`) |
|---|---|
| `load_state()` | `state.load_state()` |
| `save_state()` | `state.save_state()` |
| `sha256()` | `state.sha256()` |
| `ingest_one()` | `IngestClient.ingest()` |
| `main()` 안의 스캔·분류 loop | `core.scan_pool()` |
| `main()` 안의 실행 loop | `core.sync()` |

CLI 패키지의 `core.sync()` 함수 하나만 봐도 원시 `main()` 과 흐름이 1:1 일치한다. **함수 시그니처가 늘어난 것뿐**, 알고리즘은 그대로.

## 어느 걸 systemd 에서 부를까?

`server/systemd/rag-sync.service` 의 `ExecStart` 를 아래 둘 중 하나로:

**원시 (현재 기본):**
```ini
ExecStart=/upload/rag/venv/bin/python3 /upload/rag/scripts/rag_sync.py
```

**CLI (선택):**
```ini
ExecStart=/usr/local/bin/rag-ingest sync
```

두 방식 모두 최종적으로 같은 로직을 태우므로 결과 동일. CLI 쪽이 로그가 조금 더 예쁘게 나오고 향후 옵션 추가 (`--limit`, `--only`) 시 유연.

## 재사용 예시

CLI 패키지의 모듈은 다른 Python 코드에서 import 해서 쓸 수 있다:

```python
import pathlib
from rag_ingest.core import sync
from rag_ingest.ingest_client import IngestClient

client = IngestClient("http://rag.company.com")
result = sync(
    pool_dir=pathlib.Path("./docs"),
    state_file=pathlib.Path("./state.json"),
    client=client,
    limit=0,       # no limit
    dry_run=False,
)
print(f"Ingested {result.done}, failed {result.failed}")
```

원시 스크립트는 이런 재사용 불가 (단일 스크립트 실행 전용).

## 관련 문서

- [`docs/DESIGN_PRINCIPLES.md`](../docs/DESIGN_PRINCIPLES.md) — "Dumb Server, Smart Client" 원칙
- [`docs/ARCHITECTURE.md`](../docs/ARCHITECTURE.md) — 전체 시스템 아키텍처
- [Epic #16](https://github.com/hjoong-tangunsoft/RAG_maker/issues/16) — 자동화 파이프라인 로드맵
- [Issue #13](https://github.com/hjoong-tangunsoft/RAG_maker/issues/13) — CLI 도구 스펙
