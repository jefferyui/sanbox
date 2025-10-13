import pandas as pd
import time
from functools import wraps

def retry_df(retries=3, delay=2):
    """
    一個裝飾器：重試多次，若仍失敗則回傳空 DataFrame。
    retries: 最大重試次數
    delay: 每次重試之間的等待秒數
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(1, retries + 1):
                try:
                    df = func(*args, **kwargs)
                    
                    # 如果成功且非空 DataFrame，直接回傳
                    if isinstance(df, pd.DataFrame) and not df.empty:
                        print(f"✅ 第 {attempt} 次成功")
                        return df
                    else:
                        print(f"⚠️ 第 {attempt} 次回傳空資料，重試中...")
                
                except Exception as e:
                    print(f"❌ 第 {attempt} 次出錯: {e}")
                
                # 非最後一次才等待
                if attempt < retries:
                    time.sleep(delay)
            
            # 三次都失敗
            print("🚫 所有嘗試失敗，回傳空 DataFrame")
            return pd.DataFrame()
        return wrapper
    return decorator
import random

@retry_df(retries=3, delay=1)
def fetch_data():
    # 模擬隨機錯誤或空資料
    r = random.random()
    if r < 0.5:
        raise Exception("Network Error")
    elif r < 0.8:
        return pd.DataFrame()  # 空
    else:
        return pd.DataFrame({"a": [1, 2, 3]})  # 成功

df = fetch_data()
print(df)



#####################
import pandas as pd
import time
from functools import wraps

def retry_df(retries=3, delay=2):
    """
    裝飾器：重試多次，成功回傳 (DataFrame, True)
    全部失敗回傳 (空 DataFrame, False)
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(1, retries + 1):
                try:
                    df = func(*args, **kwargs)
                    
                    # 檢查是否為 DataFrame 且非空
                    if isinstance(df, pd.DataFrame) and not df.empty:
                        print(f"✅ 第 {attempt} 次成功")
                        return df, True
                    else:
                        print(f"⚠️ 第 {attempt} 次回傳空資料，重試中...")
                
                except Exception as e:
                    print(f"❌ 第 {attempt} 次出錯: {e}")
                
                # 若非最後一次，等待再重試
                if attempt < retries:
                    time.sleep(delay)
            
            # 若三次都失敗
            print("🚫 所有嘗試失敗，回傳空 DataFrame")
            return pd.DataFrame(), False
        return wrapper
    return decorator
import random

@retry_df(retries=3, delay=1)
def fetch_data():
    r = random.random()
    if r < 0.4:
        raise Exception("Network Error")
    elif r < 0.8:
        return pd.DataFrame()  # 空資料
    else:
        return pd.DataFrame({"value": [1, 2, 3]})  # 成功

df, success = fetch_data()
print("✅ 成功狀態:", success)
print(df)


