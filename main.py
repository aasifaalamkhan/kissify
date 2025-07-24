# main.py (Hello World Test)
import runpod
import time

def handler(job):
    # This handler just simulates work and returns a message.
    print("✅ Hello World job received:", job)
    time.sleep(5) 
    return {"output": "Hello from your test worker!"}

print("🚀 Test worker is ready and waiting for jobs.")
runpod.serverless.start({"handler": handler})
