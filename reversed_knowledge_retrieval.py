from src.predict_trigger.get_predicted_trigger import llm_attack_function

user_prompt = "USER: What is the capital of [X] ?\n ASSISTANT: "
response = " Paris"
adv_init = " country country"

print(f"The Prompt:{repr(user_prompt)}")
print(f"The response:{repr(response)}")
print(f"The initial tokens:{repr(adv_init)} is given.")

best_trigger, completion = llm_attack_function(
    num_steps=20,
    adv_string_init=adv_init,
    user_prompt=user_prompt,
    target=response,
    batch_size=128,
)
