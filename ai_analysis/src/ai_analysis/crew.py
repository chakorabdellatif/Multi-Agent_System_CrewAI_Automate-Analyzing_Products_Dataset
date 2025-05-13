from crewai import Agent, Crew, Process, Task  
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List

from src.ai_analysis.tools.DataAnalysisTool import (
    LoadDataTool,
    DataCleanerTool,
    OutlierRemoverTool
)

from src.ai_analysis.tools.pattern_analysis_tools import (
    TopRatedBrandsTool,
    AvgPriceByCategoryTool,
    TopColorPerCategoryTool,
    ProductCountByBrandTool
)

from src.ai_analysis.tools.visualization_tools import (
    BarChartAvgPriceTool,
    ScatterPlotTool,
    BarChartTopRatedBrandsTool,
    BarChartProductCountByBrandTool
)

@CrewBase
class AiAnalysis():
    """Fashion Product Analysis Crew"""

    agents: List[BaseAgent]
    tasks: List[Task]

    @agent
    def data_preparer(self) -> Agent:
        return Agent(
            config=self.agents_config['data_preparer'],
            tools=[
                LoadDataTool(),
                DataCleanerTool(),
                OutlierRemoverTool()
            ],
            verbose=True,
            memory=True  
        )

    @agent
    def pattern_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config['pattern_analyst'],
            tools=[
                TopRatedBrandsTool(),
                AvgPriceByCategoryTool(),
                TopColorPerCategoryTool(),
                ProductCountByBrandTool()
            ],
            verbose=True,
            allow_delegation=False 
        )

    @agent
    def visualization_engineer(self) -> Agent:
        return Agent(
            config=self.agents_config['visualization_engineer'],
            tools=[
                BarChartAvgPriceTool(),
                ScatterPlotTool(),
                BarChartTopRatedBrandsTool(),
                BarChartProductCountByBrandTool()
            ],
            verbose=True
        )

    @task
    def data_cleaning_task(self) -> Task:
        return Task(
            config=self.tasks_config['data_cleaning_task']
        )

    @task
    def pattern_analysis_task(self) -> Task:
        return Task(
            config=self.tasks_config['pattern_analysis_task'],
            output_dir='knowledge/patterns'
        )

    @task
    def visualization_task(self) -> Task:
        return Task(
            config=self.tasks_config['visualization_task'],
            output_dir='knowledge/plots'
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
            memory=True,
            full_output=True
        )
