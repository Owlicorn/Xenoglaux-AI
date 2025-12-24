

import sys
import subprocess
import importlib.util
import os

def check_and_install_dependencies():
    """Check and install required dependencies"""
    required_packages = [
        'torch',
        'torchvision',
        'numpy',
        'tqdm',
        'tokenizers',
        'python-dotenv',
        'pymongo',
        'requests',
        'beautifulsoup4',
        'lxml'
    ]

    # Check for torch_xla (optional, for TPU support)
    try:
        importlib.import_module('torch_xla')
    except ImportError:
        print("🔄 Installing torch_xla for TPU support...")
        try:
            subprocess.check_call([
                sys.executable, '-m', 'pip', 'install',
                'torch_xla', '--quiet'
            ])
        except subprocess.CalledProcessError:
            print("⚠️  Could not install torch_xla. TPU support will be unavailable.")

    missing_packages = []

    for package in required_packages:
        try:
            importlib.import_module(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)

    if missing_packages:
        print(f"🔄 Installing missing packages: {', '.join(missing_packages)}")
        try:
            subprocess.check_call([
                sys.executable, '-m', 'pip', 'install',
                *missing_packages, '--quiet'
            ])
            print("✅ Dependencies installed successfully!")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install dependencies: {e}")
            print("Please install manually: pip install -r requirements.txt")
            sys.exit(1)

def main():
    """Main entry point"""
    print("🦉 Xenoglaux AI - Advanced Language Model")
    print("=" * 50)

    # Check and install dependencies
    check_and_install_dependencies()

    # Import after ensuring dependencies are installed
    try:
        from trainer import XenoglauxTrainer
        from inference import XenoglauxInference
        import config
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Please check your installation.")
        sys.exit(1)

    if len(sys.argv) < 2:
        print("Usage:")
        print("  python main.py train [options]  - Train the model")
        print("  python main.py infer [options]  - Run inference")
        print("  python main.py chat             - Interactive chat mode")
        print("\nTraining options:")
        print("  --streaming    Use streaming mode for large datasets")
        print("  --epochs N     Number of epochs (default: from config)")
        print("  --batch-size N Batch size (default: from config)")
        print("  --lr N         Learning rate (default: from config)")
        print("\nInference options:")
        print("  --model PATH   Path to model file")
        print("  --tokenizer PATH Path to tokenizer")
        sys.exit(1)

    command = sys.argv[1].lower()

    if command == 'train':
        print("🚀 Starting training mode...")
        streaming = '--streaming' in sys.argv

        # Parse additional args
        epochs = None
        batch_size = None
        lr = None

        try:
            if '--epochs' in sys.argv:
                idx = sys.argv.index('--epochs')
                epochs = int(sys.argv[idx + 1])
            if '--batch-size' in sys.argv:
                idx = sys.argv.index('--batch-size')
                batch_size = int(sys.argv[idx + 1])
            if '--lr' in sys.argv:
                idx = sys.argv.index('--lr')
                lr = float(sys.argv[idx + 1])
        except (ValueError, IndexError):
            print("❌ Invalid training parameters")
            sys.exit(1)

        # Override config if specified
        conf = config.Config()
        if epochs:
            conf.EPOCHS = epochs
        if batch_size:
            conf.BATCH_SIZE = batch_size
        if lr:
            conf.LEARNING_RATE = lr

        try:
            trainer = XenoglauxTrainer(persistent_mongo=True, use_streaming=streaming)
            trainer.config = conf
            trainer.train()
        except KeyboardInterrupt:
            print("\n⚠️ Training interrupted by user")
            trainer.cleanup()
        except Exception as e:
            print(f"\n❌ Training failed: {e}")
            if 'trainer' in locals():
                trainer.cleanup()
            sys.exit(1)

    elif command == 'infer':
        print("🔍 Starting inference mode...")

        model_path = None
        tokenizer_path = None

        try:
            if '--model' in sys.argv:
                idx = sys.argv.index('--model')
                model_path = sys.argv[idx + 1]
            if '--tokenizer' in sys.argv:
                idx = sys.argv.index('--tokenizer')
                tokenizer_path = sys.argv[idx + 1]
        except IndexError:
            print("❌ Invalid inference parameters")
            sys.exit(1)

        try:
            inference = XenoglauxInference(
                model_path=model_path,
                tokenizer_path=tokenizer_path,
                persistent_mongo=True,
                use_smart_context=True,
                enable_web_search=True
            )

            if not inference._tokenizer_ready:
                print("❌ Tokenizer not ready. Please train the model first.")
                sys.exit(1)

            # Interactive inference mode
            print("💬 Interactive Inference Mode")
            print("Type 'quit' or 'exit' to stop")
            print("-" * 30)

            while True:
                try:
                    user_input = input("You: ").strip()
                    if user_input.lower() in ['quit', 'exit', 'q']:
                        break

                    if user_input:
                        response = inference.generate_response(user_input)
                        print(f"Xenoglaux: {response}")
                        print("-" * 30)

                except KeyboardInterrupt:
                    print("\n👋 Goodbye!")
                    break
                except Exception as e:
                    print(f"❌ Error: {e}")
                    continue

        except Exception as e:
            print(f"❌ Inference failed: {e}")
            sys.exit(1)

    elif command == 'chat':
        print("💬 Starting chat mode...")
        # Alias for infer command
        sys.argv[1] = 'infer'
        main()

    else:
        print(f"❌ Unknown command: {command}")
        print("Use 'train', 'infer', or 'chat'")
        sys.exit(1)

if __name__ == "__main__":
    main()