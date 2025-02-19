# **Step no. 01:** Import necessary libraries
from crewai import Crew, Task, Agent                    #library to build multi-agent
from crewai_tools import SerperDevTool                  #provides tools used by agents
from langchain_community.llms import HuggingFaceHub     #allows to integrate HuggingFace NLP models
import os                                               #to manage API keys

# **Step no. 02:** Setting up API keys (replace API_keys with your own keys)
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "<Your HUGGING_FACE API KEY HERE>"
os.environ["SERPER_API_KEY"] = "<Your HUGGING_FACE API KEY HERE>"
print("API-Keys are set successfully !!")

# **Step no. 03:** Setting up parameters for LLMS
# temperature: high_value (makes more creative and diverse response), value_range=1.0-1.5
#              low_value  (makes more predictable and deterministic response), value_range=0.1-0.3 
#              0.7 is mix value (combining creativity and accuracy) 
# max_length:  max no. of tokens(words) a model can generate
# Parameters
parameters = {"decoding_method": "greedy", "max_new_tokens": 500, "handle_parsing_errors":True, "temperature":0.7}
print("LLM parameters' values are set successfully !!")

# **Step no. 04:** Create 1st LLM using Hugging Face
llm = HuggingFaceHub(
    repo_id="meta-llama/Llama-3.3-70B-Instruct",
    model_kwargs=parameters,
)
print("1st LLM model is setup successfully!!")

#Create Function calling LLM
function_calling_llm = HuggingFaceHub(
    repo_id="mistralai/Mistral-Small-24B-Instruct-2501",
    model_kwargs=parameters,
)

# **Set no. 05:** Tools setup
search = SerperDevTool()
print("Setting up Tools sucessfully used by Agent")

# **Set no. 06:** Create Research Agent
researcher = Agent(
    llm=llm,
    function_calling_llm=function_calling_llm,
    role="Senior AI Researcher",
    goal= "Identify cutting-edge Deep Learning research in computer vision, particularly object detection and image segmentation, with a focus on practical applications.",
    backstory="You are a highly respected Deep Learning researcher with 10 years of experience in computer vision. You've published several influential papers in top conferences.",
    allow_delegation=False,    #whether the agent is allowed to delegate tasks to other agents or not
    tools=[search],
    max_iterations=50,
    max_execution_time=80.0,
    verbose=1,                 #controls the level of output(logging) that the agent provides while executing tasks i.e. 0(silent mode), 1(basic), 2(detail) 
)
print("Research Agent is created successfully")

# **Step no. 07:** Create Research Agent Task
task1 = Task(
    description="Find 3 research papers on novel applications of Deep Learning in computer vision, focusing on object detection or image segmentation.  For each paper, provide: 1) The paper title, 2) The authors, 3) A brief (2-3 sentence) summary of the key innovation, 4) A link to the paper (if available), and 5) The conference or journal where it was published.",
    expected_output="A list of 3 research papers, each with the requested information.",
    output_file="task1_output.txt",
    agent=researcher,
)
print("Task1 for Research Agent has been set successfully !!")

# **Step no. 08:** Create Writer Agent
writer = Agent(
   llm=llm,
   role="Senior Speech Writer",
   goal="Write engaging and witty keynote speeches from the provided research.",
   backstory="You are a veteran Deep Learning writer with a background in Artificial Intelligence",
   allow_delegation=False,
   verbose=1,
)
print("Writer Agent is created successfully !!")

# **Step no. 09:** Create Writer Agent Task
task2= Task(
    description="Write an engaging keynote speech on Deep Learning",
    expected_output="A detailed keynote speech with into, body and conclusion.",
    output_file="task2_output.txt",
    agent=writer,
)
print("Task2 for Writer Agent has been set successfully !!")

# **Step no. 10:** Assemble the crew
crew = Crew(agents=[researcher,writer], tasks=[task1,task2], verbose=1)
print(crew.kickoff())
