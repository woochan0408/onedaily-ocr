#!/usr/bin/env python3
import sys
import json
from datetime import datetime

def main():
    # 커맨드라인 인자 받기
    if len(sys.argv) < 2:
        result = {
            "success": False,
            "error": "No message provided"
        }
    else:
        message = sys.argv[1]
        result = {
            "success": True,
            "message": f"Hello from Python! Received: {message}",
            "timestamp": datetime.now().isoformat(),
            "pythonVersion": sys.version
        }

    # JSON 출력 (Java가 읽을 것)
    print(json.dumps(result, ensure_ascii=False))

if __name__ == "__main__":
    main()
