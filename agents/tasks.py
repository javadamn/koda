def hypothesis_workflow(question: str):
    agents = MicrobialAgents().create_agents()
    
    tasks = [
        Task(
            description=f"Analyze query: {question}",
            agent=agents["query_interpreter"],
            expected_output="Structured Cypher query parameters"
        ),
        Task(
            description="Identify key cross-feeding relationships",
            agent=agents["crossfeeding_analyst"],
            expected_output="List of significant metabolite exchanges"
        ),
        Task(
            description="Generate potential interaction hypotheses",
            agent=agents["hypothesis_generator"],
            expected_output="3 novel microbial interaction hypotheses"
        ),
        Task(
            description="Assess health implications",
            agent=agents["health_impact_analyst"],
            expected_output="Health impact analysis report"
        ),
        Task(
            description="Compile final report",
            agent=agents["report_synthesizer"],
            expected_output="Formatted Markdown report with sections"
        )
    ]
    
    return Process(tasks, verbose=True).execute()