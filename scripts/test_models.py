import sys
sys.path.append(".")

try:
    from base_retriever import SharedModels
    print("✅ base_retriever import OK")
    
    models = SharedModels()
    print("✅ Models loaded successfully")
    
except Exception as e:
    print(f"❌ Error: {e}")