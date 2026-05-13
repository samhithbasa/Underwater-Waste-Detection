import subprocess
import time
import sys

scripts = [
    'scripts/train_yolov11.py',
    'scripts/train_yolov10.py',
    'scripts/train_yolov9.py'
]

def run_master():
    print("=" * 60)
    print("🎓 MASTER TRAINING SESSION STARTED")
    print("Models will be trained sequentially while you sleep.")
    print("=" * 60)
    
    start_time = time.time()

    for script in scripts:
        print(f"\n🚀 Starting: {script}")
        try:
            # Run the script and wait for it to finish
            process = subprocess.run([sys.executable, script], check=True)
            print(f"✅ Finished: {script}")
        except subprocess.CalledProcessError as e:
            print(f"❌ Error training {script}: {e}")
            continue

    total_time = (time.time() - start_time) / 3600
    print("\n" + "=" * 60)
    print(f"🎉 ALL TRAINING COMPLETE!")
    print(f"Total time taken: {total_time:.2f} hours")
    print("You can now restart app.py to see the final results.")
    print("=" * 60)

if __name__ == '__main__':
    run_master()
