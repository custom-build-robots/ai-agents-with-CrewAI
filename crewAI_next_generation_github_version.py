# Autor:    Ingmar Stapel
# Datum:    20240330
# Version:  1.0
# Homepage: https://ai-box.eu/

import os
import json
import requests
from crewai import Agent, Task, Crew, Process
from crewai_tools import tool
import streamlit as st
import datetime
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from textwrap import dedent

# Source
# The following GitHub repo helped me alot to build this app
# URL: https://github.com/joaomdmoura/crewAI

# This video hleped me to get the streamlit agent callback functionality running
# URL: https://www.youtube.com/watch?v=nKG_kbQUDDE

# The repository from Tony Kipkemboi explains very nice how to use agents and tools.
# The SearchTools is from his repository.
#https://github.com/tonykipkemboi/trip_planner_agent


# Additional information
# Ollama as OpenSource large language model server is needed 
# to run this web-app.
# URL: https://ollama.com/

# You can choose to use a local model through Ollama for example. 
from langchain_community.llms import Ollama


# The URL below shows the API endpoint and lists all available LLMs hosted by 
# the Ollama server you are running on-prem.
# Please change the IP-address for you Ollama server.
json_url = "http://192.168.2.57:11434/api/tags"
local_base_url="http://192.168.2.57:11434"

# I have published a HowTo setup Ollama server that it works over the network
# URL: https://ai-box.eu/top-story/ollama-ubuntu-installation-und-konfiguration/1191/

# This configures the ollama_llm which will be used by our agents later.
ollama_llm = Ollama(model="openhermes", base_url=local_base_url)


# Install duckduckgo-search for this example:
# !pip install -U duckduckgo-search
from langchain_community.tools import DuckDuckGoSearchRun
search_tool = DuckDuckGoSearchRun()

from tools.search_tools import SearchTools

st.set_page_config(page_title="Your network of AI agents")

tab0, tab4, tab1, tab3, tab2 = st.tabs(["Main: ", "The tasks", "Researcher: ", "Business Angel: ", "Autor: "])

task_value_1 = "empty"
task_value_2 = "empty"
task_value_3 = "empty"
# Fetch JSON data from the URL with the model names
response = requests.get(json_url)


# This is more or less a work around that hopefully will work for the dd_search.
@tool('DuckDuckGoSearch')
def dd_search(search_query: str):
    """Search the web for information on a given topic"""
    return DuckDuckGoSearchRun().run(search_query)

# To display what the agents are currently doing this streamlit_callback function is needed.
def streamlit_callback(step_output):
    # This function will be called after each step of the agent's execution
    st.markdown("---")
    for step in step_output:
        if isinstance(step, tuple) and len(step) == 2:
            action, observation = step
            if isinstance(action, dict) and "tool" in action and "tool_input" in action and "log" in action:
                st.markdown(f"# Action")
                st.markdown(f"**Tool:** {action['tool']}")
                st.markdown(f"**Tool Input** {action['tool_input']}")
                st.markdown(f"**Log:** {action['log']}")
                st.markdown(f"**Action:** {action['Action']}")
                st.markdown(
                    f"**Action Input:** ```json\n{action['tool_input']}\n```")
            elif isinstance(action, str):
                st.markdown(f"**Action:** {action}")
            else:
                st.markdown(f"**Action:** {str(action)}")

            st.markdown(f"**Observation**")
            if isinstance(observation, str):
                observation_lines = observation.split('\n')
                for line in observation_lines:
                    if line.startswith('Title: '):
                        st.markdown(f"**Title:** {line[7:]}")
                    elif line.startswith('Link: '):
                        st.markdown(f"**Link:** {line[6:]}")
                    elif line.startswith('Snippet: '):
                        st.markdown(f"**Snippet:** {line[9:]}")
                    elif line.startswith('-'):
                        st.markdown(line)
                    else:
                        st.markdown(line)
            else:
                st.markdown(str(observation))
        else:
            st.markdown(step)

# Now set the session state for the text variables.
if "text_task_in1" not in st.session_state:
    st.session_state.text_task_in1 = None

if "text_task_in2" not in st.session_state:
    st.session_state.text_task_in2 = None

if "text_task_in3" not in st.session_state:
    st.session_state.text_task_in3 = None

# Start with the design and building up the functionality of the web-app.
# The architecture and technical design of the web-app is not very nice.
# Feel free to optimize that.
with tab1:
  st.subheader("Your research agent:")

  # Check if the request was successful
  if response.status_code == 200:
      # Parse the JSON response
      data = response.json()

      # Extract the model names from the JSON response
      names = [model["name"] for model in data["models"]]

      default_id=names.index("openhermes:latest")
      # Populate the dropdown box
      model_researcher = st.selectbox('Select a LLM model for the researcher:', names, key="model_researcher", index=default_id)
  else:
      st.error(f"Failed to fetch data from {json_url}. Error code: {response.status_code}")

  # Create a slider to select the temperature of the llm
  temperature_researcher = st.slider('Select a LLM temperature value between 0 and 1 [higher is more creative, lower is more coherent]', key="temperature_researcher", min_value=0.0, max_value=1.0, step=0.01)

  max_iterations_researcher = st.selectbox('Set the max value for interations:', ('5', '10', '15', '20', '25'), key="iter_researcher", index=2)
  ollama_llm_researcher = Ollama(model=model_researcher, base_url=local_base_url, temperature=temperature_researcher)

  role_researcher = st.text_area('role:','Senior research analyst', key="role_researcher", height=20)
  goal_researcher = st.text_area('goal:', 'As a Senior Research Analyst, you play a key role in analyzing data to offer strategic insights for decision-making. This requires strong analytical skills, critical thinking, and industry knowledge.', key="goal_researcher", height=200)
  backstory_researcher = st.text_area('backstory:', 'As a Senior Research Analyst, you hold an advanced degree in fields like economics or statistics. With expertise in research methodologies and data analysis, you execute projects across diverse industries. Your insights aid decision-making, and you stay updated on industry trends through continuous learning.', key="backstory_researcher", height=200)

with tab2:
  st.subheader("Your author agent:")

  # Check if the request was successful
  if response.status_code == 200:
      # Parse the JSON response
      data = response.json()

      # Extract the model names from the JSON response
      names = [model["name"] for model in data["models"]]

      default_id=names.index("mistral:latest")
      # Populate the dropdown box
      model_autor = st.selectbox('Select a LLM model for the autor:', names, key="model_autor", index=default_id)
  else:
      st.error(f"Failed to fetch data from {json_url}. Error code: {response.status_code}")

  # Create a slider to select the temperature of the llm
  temperature_autor = st.slider('Select a LLM temperature value between 0 and 1 [higher is more creative, lower is more coherent]', key="temperature_autor", min_value=0.0, max_value=1.0, step=0.01)

  max_iterations_autor = st.selectbox('Set the max value for interations:', ('5', '10', '15', '20', '25'), key="iter_autor", index=2)
  ollama_llm_autor = Ollama(model=model_autor, base_url=local_base_url, temperature=temperature_autor)


  role_autor = st.text_area('role:','Tech content autor', key="role_autor", height=20)
  goal_autor = st.text_area('goal:', 'As a Tech Content Author you are playing a crucial role in creating and curating high-quality content focused on technology topics. This role requires a combination of technical expertise, writing proficiency, and the ability to communicate complex concepts in a clear and engaging manner.', 
                            key="goal_autor", height=200)
  backstory_autor = st.text_area('backstory:', 'As a Tech Content Author, you hold a degree in journalism, communications, computer science, or related fields. With a passion for technology, you possess a deep understanding of technical concepts and trends. Starting your career in roles like technical writing or content creation, you have honed strong writing skills and the ability to simplify complex ideas. Through continuous learning, you stay updated on emerging technologies, ensuring your content remains relevant in the ever-changing tech landscape.', 
                                 key="backstory_autor", height=200)


with tab3:
  # This is the tab which is used to define the agent specifig llm agent.
  # All the description below is used as an example that the user of that web-app
  # has an idea how to define such an agent.
  st.subheader("Your investor agent:")

  # Check if the request was successful an the ollama server is responding.
  if response.status_code == 200:
      # Parse the JSON response
      data = response.json()

      # Extract the model names from the JSON response generated by the ollama server
      names = [model["name"] for model in data["models"]]

      # Populate the dropdown box with the available models. Set openhermes as default.
      default_id=names.index("openhermes:latest")
      model_consultant = st.selectbox('Select a LLM model for the agent:', names, key="model_consultant", index=default_id)
  else:
      st.error(f"Failed to fetch data from {json_url}. Error code: {response.status_code}")

  # Create a slider to select the temperature of the llm
  temperature_consultant = st.slider('Select a LLM temperature value between 0 and 1 [higher is more creative, lower is more coherent]', key="temperature_consultant", min_value=0.0, max_value=1.0, step=0.01)

  # Set the max value how long an agent is allowed to interate.
  max_iterations_consultant = st.selectbox('Set the max value for interations:', ('5', '10', '15', '20', '25'), key="iter_consultant", index=2)
  
  # Define the llm call for the ollama server we like to use for our agent  
  ollama_llm_consultant = Ollama(model=model_consultant, base_url=local_base_url, temperature=temperature_consultant)

  # Define now our agent
  role_consultant = st.text_area('role:','Business Angel and venture capital consultant', key="role_consultant", height=20)
  goal_consultant = st.text_area('goal:', 'As a Business Angels and Venture Capital Consultant you are playing a vital role in the startup ecosystem by providing funding, mentorship, and strategic guidance to early-stage companies. While their roles share similarities, they differ in terms of investment focus, funding sources, and level of involvement.', 
                key="goal_consultant", height=200)
  backstory_consultant = st.text_area('backstory:', 'Business Angels and Venture Capital Consultants typically possess extensive experience in finance, entrepreneurship, and investment management. They may have backgrounds in fields such as investment banking, private equity, corporate finance, or startup leadership. Many have built successful careers in the financial industry, gaining expertise in deal sourcing, due diligence, portfolio management, and strategic advisory.', 
                                      key="backstory_consultant", height=200)

with tab4:
  st.subheader("The agent tasks:")

  st.session_state.text_task_in1 = st.text_area('Task 1 Researcher:', 
                                                dedent(f"""Conduct a comprehensive analysis of the latest high performing startups active in the 
field of generative AI. It is important that those startups with their advancements in 
generative AI are active in the finance sector since a year. Identify key startups, 
breakthrough technologies, and potential fast growing startups with impact in the finance 
sector caused by generative AI. As a researcher you analyse how generative AI will change 
the finance industry. It would be good to know if that startup is still searching for money 
investments actively. Your final answer MUST be a full analysis report.
Example Report: 
    Finance Tech Startup Research Table: 
    - Startup 1: 
        - Name: "Kern AI" 
        - Investment sum: 1.00.00.000 
        - Founded in: 2022 
        - Number of Employees: 50 
        - Company homepage: https://www.kern.ai/ 
    - Startup 2: 
        - Name: "Scrub AI" 
        - Investment sum: 5.00.00.000 
        - Founded in: 2023 
        - Number of Employees: 22 
        - Company homepage: https://scrub-ai.com/
Today is the """)+str(datetime.date.today())+""" .""", key="text_task_in_1")

  st.session_state.text_task_in2 = st.text_area('Task 2 Autor / Writer:',
                                                dedent(f"""Using the insights provided, write an article like an engaging blog post that highlights the most significant startups 
active in generative AI with important advancements in this field. Your written article should be informative yet accessible, catering to a tech-savvy startup scene and 
audience. Make it sound cool, avoid complex words so it doesn't sound like AI. Your final answer MUST be the a full structures blog post 
The article you are writing has a minimum of 1600 words and highlights 10 startups. In the summary please list the startups with web addresses like url's headlines and bullet points for easy reading. 
The text itself is enriched with nice emojis to highlight important parts.
                                                       
The structure of the article you have to write could look like the example below: 
                                                       
Example article structure:
    Executive Summary: 
    - Overview of the AI startup's performance. 
    - Key financial metrics and achievements. 
    - Future growth prospects. 
    - Introduction: 
    - Brief background of the AI startup. 
    - Mission and objectives. 
    - Market Analysis: 
    - Analysis of the AI market segment. 
    - Growth trends and opportunities. 
    - Competitive landscape. 
    - Business Model: 
        - Description of the AI startup's business model. 
        - Revenue streams. 
        - Cost structure. 
    - Financial Performance: 
        - Revenue analysis: 
        - Revenue growth over time. 
    - Revenue sources (e.g., product sales, subscriptions, services). 
        - Profitability analysis: 
        - Gross profit margin. 
        - Operating profit margin. 
        - Net profit margin. 
    - Cash flow analysis: 
        - Operating cash flow. 
        - Investing cash flow. 
        - Financing cash flow. 
    - Balance sheet analysis: 
        - Assets composition. 
        - Liabilities and equity. 
    - Key financial ratios: 
        - Return on Investment (ROI). 
        - Return on Equity (ROE). 
        - Debt-to-Equity ratio. 
        - Current ratio. 
        - Quick ratio. 
    - Investment Analysis: 
        - Valuation: 
            - Methods used (e.g., Discounted Cash Flow, Comparable Company Analysis). 
            - Assumptions and inputs. 
    - Investment risks: 
        - Market risks. 
        - Technology risks. 
        - Regulatory risks. 
    - Strategic Initiatives: 
        - Expansion plans. 
        - Research and development efforts. 
        - Strategic partnerships. 
    - Conclusion: 
        - Summary of key findings. 
        - Recommendations for investors. 
        - Future outlook. 
    - Appendix: 
        - Detailed financial tables. 
        - Glossary of financial terms. 
        - References: 
    - Sources of information used in the report. \nToday is the: """) +str(datetime.date.today())+""" .""", key="text_task_in_2")

  st.session_state.text_task_in3 = st.text_area('Task 3 Business Angel:', dedent(f"""Involve evaluating investment opportunities, conducting due diligence 
on potential ventures, and advising startups on strategy, fundraising, and growth tactics. Search how much venture capital each startup already raised. 
Add a comment if an future investment would be an option for an investor. Only from interest are startups in finance sector which are active over the last 
year and this year. Additionally, they often facilitate connections between entrepreneurs and potential investors, leveraging their network to bridge the 
gap between promising startups and capital sources. 
Executive Summary:
- Concise overview of the investment opportunity.
- Highlights of key figures and decision points.
- Summary of investment recommendations.
Introduction:
- Introduction to the company or opportunity being presented.
- Purpose of the report.
- Scope and methodology.
Market Analysis:
    - Market overview:
        - Size, growth rate, and trends.
        - Market segmentation.
    - Competitive landscape:
        - Major players and market share.
        - Competitive advantages of the company.
Business Model:
- Description of the company's business model.
- Revenue streams and sources.
- Cost structure and scalability.
Financial Performance:
    - Revenue analysis:
        - Historical revenue trends.
        - Forecasted revenue growth.
    - Profitability analysis:
        - Gross margin, operating margin, net margin.
    - Cash flow analysis:
        - Operating cash flow, free cash flow.
    - Key financial ratios:
        - Return on Investment (ROI), Return on Equity (ROE), Debt-to-Equity ratio, etc.
Investment Thesis:

Investment opportunity:
    - Value proposition.
    - Unique selling points.
    - Potential returns:
        - Expected ROI.
        - Risk-adjusted returns.
    - Risks and Mitigation Strategies:

    - Identification of potential risks:
        - Market risks, operational risks, regulatory risks, etc.
    - Mitigation strategies:
        - Plans to address identified risks.
    - Strategic Growth Initiatives:
Expansion plans:
    - Geographic expansion, product diversification, etc.
Research and development:
    - Innovation pipeline and investments.
Strategic partnerships:
    - Alliances, joint ventures, collaborations.
Valuation:
    Valuation methodology:
        - Discounted Cash Flow (DCF), Comparable Company Analysis (CCA), etc.
        - Valuation assumptions and inputs.
Investment Recommendations:
    - Summary of key findings and analysis.
Investment decision:
    - Buy, sell, hold recommendations.
    - Justification of recommendations.
Conclusion:
    - Summary of the investment opportunity.
    - Closing remarks.
Appendix:
    - Detailed financial tables.
    - Glossary of financial terms.
    - Assumptions used in the analysis.
References:
- Sources of information used in the report.                          
Today is the """)+str(datetime.date.today())+""" .""", key="text_task_in_3")

  task_in_1_new = st.session_state.text_task_in1
  task_in_2_new = st.session_state.text_task_in2
  task_in_3_new = st.session_state.text_task_in3 

with tab0:
  st.title('Do my analysis')
  task_description = st.text_area('Your short task description here is used to re-write Task 1 - Task 3 so that they fit thematically with the new input.') 

  # Check if the request was successful
  if response.status_code == 200:
      # Parse the JSON response
      data = response.json()

      # Extract the model names from the JSON response
      names = [model["name"] for model in data["models"]]

      # Populate the dropdown box
      default_id=names.index("openhermes:latest")
      model_rewrite = st.selectbox('Select a LLM model for re-writing the tasks 1 - 3:', names, key="model_rewrite", index=default_id)
  else:
      st.error(f"Failed to fetch data from {json_url}. Error code: {response.status_code}")
  # Create a slider to select the temperature of the llm
  temperature_rewrite_task = st.slider('Select a LLM temperature value between 0 and 1 [higher is more creative, lower is more coherent]', min_value=0.0, max_value=1.0, step=0.01)

  if st.button('Start Generation NOW'):
    with st.status("🤖 **Now rewriting the tasks for your three agents...**", state="running", expanded=True) as status:
          ollama_llm_rewrite_task = Ollama(model=model_rewrite, base_url=local_base_url, temperature=temperature_rewrite_task)

          template_task_1 = "As an AI assistant please write a task description for an AI agent whos role is to be a researcher who like to understand various topics. This is an example task description for an AI agent. The AI agent needs this task to understand what he has to do. \n Example task description:\n" + st.session_state.text_task_in1 + "\n Please rewrite this task description for the new topic which is described as follows: \n New topic: \n{task_description} \nImportant for the rewritten new task description is to keep the structure of the example task description provided."
          prompt_task_1 = PromptTemplate(template=template_task_1, input_variables=["task_description"])
          llm_chain = LLMChain(prompt=prompt_task_1, llm=ollama_llm_rewrite_task)
          task_in_1_new = llm_chain.run({"task_description": task_description})

          template_task_3 = "As an AI assistant please write a task description for an AI agent whos role is an business angle investor who does analysis. This is an example task description for an AI agent. The AI agent needs this task to understand what he has to do. \n Example task description:\n" + st.session_state.text_task_in3 + "\n Please rewrite this task description for the new topic which is described as follows: \n New topic: \n{task_description} \nImportant for the rewritten new task description is to keep the structure of the example task description provided."
          prompt_task_3 = PromptTemplate(template=template_task_3, input_variables=["task_description"])
          llm_chain = LLMChain(prompt=prompt_task_3, llm=ollama_llm_rewrite_task)
          task_in_3_new = llm_chain.run({"task_description": task_description})

          template_task_2 = "As an AI assistant please write a task description for an AI agent whos role is to be an autor who likes to write articles. This is an example task description for an AI agent. The AI agent needs this task to understand what he has to do. \n Example task description:\n" + st.session_state.text_task_in2 + "\n Please rewrite this task description for the new topic which is described as follows: \n New topic: \n{task_description} \nImportant for the rewritten new task description is to keep the structure of the example task description provided."
          prompt_task_2 = PromptTemplate(template=template_task_2, input_variables=["task_description"])
          llm_chain = LLMChain(prompt=prompt_task_2, llm=ollama_llm_rewrite_task)
          task_in_2_new = llm_chain.run({"task_description": task_description})


          st.text_area('Task 1 Researcher rewritten:', task_in_1_new, key="text_task_in_1_re")
          st.text_area('Task 3 Business Angel rewritten:', task_in_3_new, key="text_task_in_3_re")
          st.text_area('Task 2 Autor / Writer rewritten:', task_in_2_new, key="text_task_in_2_re")

    # Define your agents with roles and goals
    researcher = Agent(
      max_inter=max_iterations_researcher,
      role=role_researcher,
      goal=goal_researcher,
      backstory=backstory_researcher,

      verbose=True,
      allow_delegation=True,
      tools=[
          SearchTools.search_internet,
          dd_search,
      ],
      llm=ollama_llm_researcher, 
      step_callback=streamlit_callback
    )

    consultant = Agent(
      max_inter=max_iterations_consultant,
      role=role_consultant,
      goal=goal_consultant,
      backstory=backstory_consultant,

      verbose=True,
      allow_delegation=False,
      tools=[
          SearchTools.search_internet,
          dd_search,
      ],
      llm=ollama_llm_consultant, 
      step_callback=streamlit_callback
    )

    autor = Agent(
      max_inter=max_iterations_autor,
      role=role_autor ,
      goal=goal_autor,
      backstory=backstory_autor,
      verbose=True,
      allow_delegation=False,
      llm=ollama_llm_autor,
      step_callback=streamlit_callback
    )

    # Create tasks for your agents
    task1 = Task(
      description=task_in_1_new,
      agent=researcher,
      expected_output="Do my work please"
    )

    # Create tasks for your agents
    task2 = Task(
      description=task_in_2_new,
      agent=autor,
      expected_output="Do my work please"
    )

    # Create tasks for your agents
    task3 = Task(
      description=task_in_3_new,
      agent=consultant,
      expected_output="Do my work please"
    )

    with st.status("🤖 **Agents doing your work...**", state="running", expanded=True) as status:
        with st.container(height=800, border=False):
          crew = Crew(
            agents=[researcher, consultant, autor],
            tasks=[task1, task3, task2],
            verbose=2, # You can set it to 1 or 2 to different logging levels
          )
          result = crew.kickoff()
        status.update(label="✅ Research activity finished!",
                      state="complete", expanded=False)

    print("######################")
    print(result)
    st.subheader('Your requested analysis is ready: :blue[how cool is that] :sunglasses:')
    st.markdown(result)

    st.download_button(
        label="Download",
        data=result, 
        file_name="meeting_prep.md",
        mime="text/plain"
    )
