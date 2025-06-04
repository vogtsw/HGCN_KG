import torch

def check_model(model_path):
    try:
        state_dict = torch.load(model_path, map_location='cpu')
        print("Model keys:", state_dict.keys() if isinstance(state_dict, dict) else "Not a dictionary")
        if isinstance(state_dict, dict):
            for key in state_dict.keys():
                if isinstance(state_dict[key], dict):
                    print(f"\n{key} keys:", state_dict[key].keys())
    except Exception as e:
        print("Error loading model:", str(e))

if __name__ == "__main__":
    model_path = "checkpoints/entity_enhanced/best_model.pt"
    check_model(model_path) 