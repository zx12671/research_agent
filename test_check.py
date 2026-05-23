import sys
print(f"Python version: {sys.version}")

try:
    import openai
    print(f"openai OK (version: {openai.__version__})")
except Exception as e:
    print(f"openai FAIL: {e}")

try:
    import torch
    print(f"torch OK (version: {torch.__version__})")
except Exception as e:
    print(f"torch FAIL: {e}")

try:
    from FlagEmbedding import BGEM3FlagModel
    print("FlagEmbedding OK")
except Exception as e:
    print(f"FlagEmbedding FAIL: {e}")

try:
    import transformers
    print(f"transformers OK (version: {transformers.__version__})")
except Exception as e:
    print(f"transformers FAIL: {e}")

try:
    import pandas
    print(f"pandas OK (version: {pandas.__version__})")
except Exception as e:
    print(f"pandas FAIL: {e}")

try:
    from sentence_transformers import SentenceTransformer
    print("sentence_transformers OK")
except Exception as e:
    print(f"sentence_transformers FAIL: {e}")
