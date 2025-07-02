# CoTGenerator

After demonstrating a task using AgentNet Tool and did preprocessing, we can use a generator foundtion model, e.g. Claude 3.7 to synthesize the reasoning part between an observation and an action. 

## How to generator CoT for your annotations
1.  Use 'gen_cot.py' to synthesize CoT

a. Choose your model and set API key: export API_KEY=xxx

b. Run gen_cot.py

2. Use CoTGenerator/visualization/app.py to visualize the results

3. (Optional) use merge_to_jsonl.py to combine all the trajectories into a single jsonl file.