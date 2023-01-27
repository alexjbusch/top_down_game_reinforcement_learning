from src.top_down import GameState
import torch


env = GameState()
model = torch.load(f"trained_models/topdown_model_44000")


angle_dict = {}

angle = 0
while angle < 360:
    env.player.angle = angle
    next_steps = env.get_next_states()
    # Exploration or exploitation

    next_actions, next_states = zip(*next_steps.items())
    #print(next_states)        
    next_states = torch.stack(next_states)
    if torch.cuda.is_available():
        next_states = next_states.cuda()
    model.eval()
    with torch.no_grad():
        predictions = model(next_states)
        action = torch.argmax(predictions).item()


    angle_dict[angle] = action
    angle += 1
    



env.visualize(angle_dict)
env.display()

