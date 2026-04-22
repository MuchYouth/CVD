# modules/ai — LLM 클라이언트 모듈

여러 LLM 제공자(Claude, ChatGPT, Gemini, Grok)를 **동일한 인터페이스**로 사용할 수 있게 추상화한 모듈입니다.

---

## 파일 구성

| 파일 | 역할 |
|------|------|
| `base.py` | `LLMClient` 추상 기본 클래스 (`generate_text` 인터페이스 정의) |
| `factory.py` | `get_llm_client()` — 환경변수에 따라 클라이언트 인스턴스 반환 |
| `claude.py` | Anthropic Claude 구현체 |
| `chatgpt.py` | OpenAI ChatGPT 구현체 |
| `gemini.py` | Google Gemini 구현체 |
| `grok.py` | xAI Grok 구현체 |

---

## 환경변수 설정 (`.env`)

```dotenv
# 사용할 LLM 제공자 (기본값: claude)
LLM_PROVIDER=claude   # claude | chatgpt | gemini | grok

# 각 제공자별 API 키 (사용하는 제공자의 키만 필수)
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
GOOGLE_API_KEY=AIza...
XAI_API_KEY=xai-...

# 모델명 (생략 시 아래 기본값 사용)
CLAUDE_MODEL=claude-sonnet-4-20250514
OPENAI_MODEL=gpt-4o
GEMINI_MODEL=gemini-2.0-flash-exp
GROK_MODEL=grok-beta
```

---

## 기본 사용법

```python
from modules.ai.factory import get_llm_client

# .env의 LLM_PROVIDER 값에 따라 자동 선택
client = get_llm_client()

response = client.generate_text(
    prompt="공격 시나리오를 설명해줘",
    system_prompt="당신은 보안 전문가입니다."  # 선택사항
)
print(response)
```

### 제공자 직접 지정

```python
client = get_llm_client(provider="gemini")
response = client.generate_text(prompt="...")
```

지원되는 `provider` 값:

| 제공자 | 허용 값 |
|--------|---------|
| Claude | `"claude"` |
| ChatGPT | `"chatgpt"`, `"openai"`, `"gpt"` |
| Gemini | `"gemini"`, `"google"` |
| Grok | `"grok"`, `"xai"` |

---

## 새 LLM 제공자 추가 방법

1. `base.py`의 `LLMClient`를 상속한 클래스를 새 파일로 작성
2. `generate_text()` 메서드 구현
3. `factory.py`의 `get_llm_client()`에 분기 추가

```python
# 예: modules/ai/newprovider.py
from .base import LLMClient

class NewProviderClient(LLMClient):
    def generate_text(self, prompt, system_prompt=None, max_tokens=4096):
        # API 호출 로직
        ...
```

```python
# factory.py에 추가
elif provider_lower == "newprovider":
    return NewProviderClient()
```

---

## 메트릭 추적

모든 구현체는 API 호출 시 `modules.core.metrics`의 `MetricsTracker`를 통해 **입출력 토큰 사용량**을 자동으로 기록합니다. 별도 설정 없이 동작합니다.
