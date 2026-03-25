import os 

from langchain_core.prompts.chat import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder, SystemMessagePromptTemplate
from src.utils import PlanExecState, PlanSummaryState, StepTaskState
from src.prompt_template import step_system_message, step_human_message, step_input_variables, summary_system_message, summary_human_message, summary_input_variables
from src.utils import StepTaskFormat, PlanSummaryFormat

from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI

load_dotenv()

def task_define(state: PlanExecState):
    API_KEY = os.getenv("API_KEY")
    messages = [
        SystemMessagePromptTemplate.from_template(step_system_message),
        HumanMessagePromptTemplate.from_template(step_human_message),
    ]
    prompt = ChatPromptTemplate(input_variables=step_input_variables, messages=messages)
    llm = ChatOpenAI(model_name=os.getenv("MODEL_NAME"), temperature=0.3, api_key=API_KEY, max_retries=5)
    structured_llm = llm.with_structured_output(StepTaskFormat)
    chain = prompt | structured_llm

    # check stop or continue
    if len(state["step_output"]) == len(state["plan"]) or (len(state["step_output"]) > 0 and state["step_output"][-1]["success"].lower() == "no"):
        # summary about this plan and then stop
        messages = [
            SystemMessagePromptTemplate.from_template(summary_system_message),
            HumanMessagePromptTemplate.from_template(summary_human_message),
        ]
        prompt = ChatPromptTemplate(input_variables=summary_input_variables, messages=messages)
        llm = ChatOpenAI(model_name=os.getenv("MODEL_NAME"), temperature=0.0, api_key=API_KEY,  max_retries=5)
        structured_llm = llm.with_structured_output(PlanSummaryFormat)

        chain = prompt | structured_llm
        question = state["original_question"]
        plan = f"[{', '.join(state['plan'])}]"
        memory = ""
        for idx, item in enumerate(state["step_output"]):
            step = state["plan"][idx]
            task = state["step_question"][idx]["task"]
            answer = item["answer"]
            rating = item["rating"]
            memory += (
                f"Task: {step}\n"
                f"Question: {task}\n"
                f"Answer: {answer}\n"
                f"Confident score: {rating}\n\n"
            )
        full_prompt = prompt.format(
            question = question,
            plan = plan,
            memory = memory
        )
        output = chain.invoke({
            "question": question, 
            "plan": plan,
            "memory": memory
        })
        output = PlanSummaryState(**output.model_dump())
        return {"plan_summary": output, "stop": True}
    else:
        plan = f"[{', '.join(state['plan'])}]"
        cur_step = state["plan"][len(state["step_output"])]
        memory = ""
        for idx in range(len(state["step_output"])):
            step = state["plan"][idx]
            answer = state["step_output"][idx]["answer"]
            memory += f"Task: {step}\nAnswer: {answer}\n\n"
        response = chain.invoke({"plan": plan, "cur_step": cur_step, "memory": memory})
        response = StepTaskState(**response.model_dump())
        return {"step_question": [response]}