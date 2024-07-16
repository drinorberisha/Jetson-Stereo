import os

def run_script(script_name):
    os.system(f"python2.7 {script_name}")

def main():
    print("Starting process...")
    run_script('scripts/LiveCapture.py')
    run_script('scripts/JulyCal.py')
    # run_script('scripts/AdaptiveHistogramEqualization.py')
    # run_script('scripts/UndistortImages.py')
    # run_script('scripts/Rectification.py')
    # run_script('scripts/sgmb3Cuda.py')
    print("Process completed!")

if __name__ == "__main__":
    main()
