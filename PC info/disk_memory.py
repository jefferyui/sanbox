import shutil
import psutil

# 查看磁碟空間
def print_disk_usage(path="/"):
    total, used, free = shutil.disk_usage(path)
    print("🖴 磁碟空間使用狀況：")
    print(f"  總容量:   {total / (1024**3):.2f} GB")
    print(f"  已使用:   {used / (1024**3):.2f} GB")
    print(f"  剩餘空間: {free / (1024**3):.2f} GB\n")

# 查看記憶體使用狀況
def print_memory_usage():
    mem = psutil.virtual_memory()
    print("💾 記憶體使用狀況：")
    print(f"  總容量:   {mem.total / (1024**3):.2f} GB")
    print(f"  已使用:   {mem.used / (1024**3):.2f} GB")
    print(f"  可用記憶體: {mem.available / (1024**3):.2f} GB")
    print(f"  使用率:   {mem.percent}%")

# 執行
print_disk_usage("/")
print_memory_usage()


