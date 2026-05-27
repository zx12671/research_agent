import torch 
import ollama
import os
from openai import OpenAI
import requests  # 添加 HTTP 请求库
import google.generativeai as genai




class GPTs:
    def __init__(self, llm_keys: str = os.environ.get("OPEN_API_KEY"), model_name: str = "gpt-4o-mini"):
        # Initialize OpenAI client with the API key and model name

        self.client = OpenAI(api_key=llm_keys)
        self.model_name = model_name


    def chat(self, message: str, conversation_history: list = None):
        try:#add error handling
            if conversation_history is None:
                conversation_history = [{"role": "system", "content": "You are a helpful assistant."}]

            # Add the user's message to the conversation history
            conversation_history.append({"role": "user", "content": message})
            # Call the OpenAI API to generate a response
            response = self.client.chat.completions.create(model=self.model_name, messages=conversation_history)

            # Get the assistant's reply from the response
            assistant_reply = response.choices[0].message.content

            # Append the assistant's reply to the conversation history
            conversation_history.append({"role": "assistant", "content": assistant_reply})

            return assistant_reply, conversation_history

        except Exception as e:
            if "context length" in str(e).lower():
                return "Error: Input exceeds model's context limit.", conversation_history
            elif "authentication" in str(e).lower():
                raise PermissionError("Invalid API key") from e
            else:
                raise e
# 定义 QianWen 类，用于调用千问模型
class QianWen:
    def __init__(self, llm_key: str = os.environ.get("QIANWEN_KEY"), model_name: str = "qwen-turbo", api_base: str = "https://dashscope.aliyuncs.com/api/v1"):
        """
        初始化千问客户端
        Args:
            llm_key: 阿里云千问 API Key
            model_name: 模型名称，如 "qwen-turbo", "qwen-plus", "qwen-max"
            api_base: API 基础 URL
        """
        self.api_key = llm_key
        self.model_name = model_name
        self.api_base = api_base
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    def chat(self, message: str, conversation_history: list = None):
        try:
            if conversation_history is None:
                conversation_history = [{"role": "system", "content": "You are a helpful assistant."}]
            
            # 添加用户消息到历史
            conversation_history.append({"role": "user", "content": message})
            
            # 调用千问 API
            url = f"{self.api_base}/chat/completions"
            payload = {
                "model": self.model_name,
                "messages": conversation_history,
                "temperature": 0.7
            }
            
            response = requests.post(url, json=payload, headers=self.headers, timeout=60)
            response.raise_for_status()
            
            result = response.json()
            
            # 检查响应格式
            if "choices" not in result or len(result["choices"]) == 0:
                raise ValueError(f"Invalid response from QianWen API: {result}")
            
            assistant_reply = result["choices"][0]["message"]["content"]
            
            # 添加助手回复到历史
            conversation_history.append({"role": "assistant", "content": assistant_reply})
            
            return assistant_reply, conversation_history
            
        except requests.exceptions.Timeout:
            raise TimeoutError("QianWen API request timed out")
        except requests.exceptions.ConnectionError:
            raise ConnectionError("Failed to connect to QianWen API")
        except Exception as e:
            if "context length" in str(e).lower():
                return "Error: Input exceeds model's context limit.", conversation_history
            elif "authentication" in str(e).lower() or "invalid" in str(e).lower():
                raise PermissionError("Invalid QianWen API key") from e
            else:
                raise e

# 定义 DeepSeek 类，用于调用 DeepSeek API
class DeepSeek:
    def __init__(self, llm_keys: str = os.environ.get("DEEPSEEK_API_KEY"), model_name: str = "deepseek-chat"):
        """
        初始化 DeepSeek 客户端
        Args:
            llm_keys: DeepSeek API Key
            model_name: 模型名称，如 "deepseek-chat", "deepseek-reasoner"
        """
        import httpx
        # 使用 httpx 客户端并设置超时，避免创建客户端时卡住
        http_client = httpx.Client(timeout=httpx.Timeout(60.0, connect=10.0))
        self.client = OpenAI(api_key=llm_keys, base_url="https://api.deepseek.com", http_client=http_client)
        self.model_name = model_name

    def chat(self, message: str, conversation_history: list = None):
        try:
            if conversation_history is None:
                conversation_history = [{"role": "system", "content": "You are a helpful assistant."}]

            # 添加用户消息到历史
            conversation_history.append({"role": "user", "content": message})

            # 调用 DeepSeek API
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=conversation_history
            )

            # 获取助手回复
            assistant_reply = response.choices[0].message.content

            # 添加助手回复到历史
            conversation_history.append({"role": "assistant", "content": assistant_reply})

            return assistant_reply, conversation_history

        except Exception as e:
            if "context length" in str(e).lower():
                return "Error: Input exceeds model's context limit.", conversation_history
            elif "authentication" in str(e).lower() or "invalid" in str(e).lower():
                raise PermissionError("Invalid DeepSeek API key") from e
            else:
                raise e

# 定义 Gemini 类，用于调用 Gemini 模型
class Gemini:
    def __init__(self, llm_key=os.environ.get("GEMINI_KEY") , model_name: str = "gemini-1.5-pro"):
        genai.configure(api_key=llm_key)
        self.model = genai.GenerativeModel(model_name)

    def chat(self, message: str, conversation_history: list = None):
        try:#add error handling
            if conversation_history is None:
                conversation_history = []
            chatone = self.model.start_chat(history=conversation_history)
            response = chatone.send_message(message)

            # Get the assistant's reply from the response
            assistant_reply = response.text

            # Append the assistant's reply to the conversation history
            conversation_history.append({"role": "model", "parts": assistant_reply})

            return assistant_reply, conversation_history

        except Exception as e:
            if "context length" in str(e).lower():
                return "Error: Input exceeds model's context limit.", conversation_history
            elif "authentication" in str(e).lower():
                raise PermissionError("Invalid API key") from e
            else:
                raise e
# class chatllms:
#     def __init__(self, model_name="Qwen", llm_keys=None) -> None:
#         self.model_name = model_name.lower()
#         self.tokenizer = None
#         self.model = None
#         #add error handling
#         if 'qwen' in self.model_name or 'llama' in self.model_name:
#             #self._load_local_model()
#             pass
#         elif "gemini" in model_name:
#             assert llm_keys is not None, "API key is required for Gemini."
#             self.model = Gemini(llm_key=llm_keys, model_name=self.model_name)
#         elif "gpt" in model_name or "o1" in model_name:
#             assert llm_keys is not None, "API key is required for GPT."
#             self.model = GPTs(llm_keys=llm_keys, model_name=self.model_name)
#         else:
#             raise ValueError(f"Unsupported model name: {self.model_name}")

class chatllms:
    def __init__(self, model_name="Qwen", llm_keys=None, api_base="https://maas-api.cn-huabei-1.xf-yun.com/v2") -> None:
        self.model_name = model_name.lower()
        self.tokenizer = None
        self.model = None
        # DeepSeek 模型（deepseek-chat, deepseek-reasoner 等）
        if 'deepseek' in self.model_name:
            assert llm_keys is not None, "API key is required for DeepSeek."
            self.model = DeepSeek(llm_keys=llm_keys, model_name=self.model_name)
        # 千问 API 模型（qwen-turbo, qwen-plus, qwen-max 等）
        elif 'qwen' in self.model_name and ('api' in self.model_name or 'turbo' in self.model_name or 'plus' in self.model_name or 'max' in self.model_name):
            self.model = QianWen(llm_key=llm_keys, model_name=self.model_name, api_base=api_base)
        elif 'qwen' in self.model_name or 'llama' in self.model_name:
            # 本地模型通过 Ollama 调用，无需创建实例（_chat_local 中处理）
            pass
        elif "gemini" in self.model_name:
            assert llm_keys is not None, "API key is required for Gemini."
            self.model = Gemini(llm_key=llm_keys, model_name=self.model_name)
        elif "gpt" in self.model_name or "o1" in self.model_name:
            assert llm_keys is not None, "API key is required for GPT."
            self.model = GPTs(llm_keys=llm_keys, model_name=self.model_name)
        else:
            raise ValueError(f"Unsupported model name: {self.model_name}")
    # @torch.no_grad()
    # def chat(self, prompt="", history=None):
    #     if 'qwen' in self.model_name or 'llama' in self.model_name:
    #         return self._chat_local(prompt, history, self.model_name)
    #     elif "gemini" in self.model_name or "gpt" in self.model_name or "o1" in self.model_name:
    #         return self.model.chat(prompt, history)
    #     else:
    #         raise ValueError(f"Unsupported model name: {self.model_name}")
    @torch.no_grad()
    def chat(self, prompt="", history=None):
        if 'deepseek' in self.model_name:
            return self.model.chat(prompt, history)
        elif 'qwen' in self.model_name:
            if 'api' in self.model_name or 'turbo' in self.model_name or 'plus' in self.model_name or 'max' in self.model_name:
                # 千问 API
                return self.model.chat(prompt, history)
            else:
                # 本地千问模型
                return self._chat_local(prompt, history, self.model_name)
        elif 'llama' in self.model_name:
            return self._chat_local(prompt, history, self.model_name)
        elif "gemini" in self.model_name or "gpt" in self.model_name or "o1" in self.model_name:
            return self.model.chat(prompt, history)
        else:
            raise ValueError(f"Unsupported model name: {self.model_name}")
    def _chat_local(self, prompt, history=None, model_name=None):
        try:#add error handling
            if history is None:
                history = []
            history.append({"role": "user", "content": prompt})
            response = ollama.chat(model=model_name, messages=history)
            assistant_message = response['message']['content']
            history.append({"role": "assistant", "content": assistant_message})
            return assistant_message, history
        except KeyError:
            raise ValueError("Invalid response format from local model")
        except Exception as e:
            print(f"Local model error: {e}")
            raise
