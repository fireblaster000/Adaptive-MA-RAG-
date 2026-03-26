from src.utils import GraphState
from src.prompt_template import planing_system_message, planing_human_message, planing_input_variables
from src.utils import PlanFormat

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import os

from langchain_core.prompts.chat import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from src.llm_profile import profile_llm_call

load_dotenv()

def plan_agent(state: GraphState):
    API_KEY = os.getenv("API_KEY")
    original_question = state["original_question"]
    all_mem = []
    for past_exp in state["past_exp"]:
        memory = ""
        plan = ", ".join(past_exp["plan"])
        memory += f"Plan: [{plan}]\n"
        summary = past_exp["plan_summary"]
        memory += f"Status: {summary['output']} Score: {summary['score']}\n"
        all_mem.append(memory)
    memory = ""
    if len(all_mem) == 0:
        memory = "empty"
    else:
        for idx in range(len(all_mem)):
            memory += f"Trial {idx}:\n{all_mem[idx]}\n"
    
    messages = [
        SystemMessagePromptTemplate.from_template(planing_system_message),
        HumanMessagePromptTemplate.from_template(planing_human_message),
    ]
    prompt = ChatPromptTemplate(input_variables=planing_input_variables, messages=messages)
    llm = ChatOpenAI(model_name=os.getenv("MODEL_NAME"), temperature=0.3, api_key=API_KEY)
    structured_llm = llm.with_structured_output(PlanFormat)
    chain = prompt | structured_llm
    def _call():
        return chain.invoke({
            "question": original_question,
            "memory": memory
        })

    output, metric = profile_llm_call(_call, stage="plan_agent")
    return {"plan": output.step, "llm_metrics": [metric]}
