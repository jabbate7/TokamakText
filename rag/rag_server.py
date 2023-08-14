from flask import Flask, render_template_string, request
from rag import rag_query
from llm_interface import OpenAIInterface, HuggingFaceInterface
import click
from typing import List

def get_llm_interface(model_name):
    if model_name == "openai":
        return OpenAIInterface()
    elif model_name == "huggingface":
        return HuggingFaceInterface()
    else:
        raise ValueError(f"Invalid model name: {model_name}")

def format_passes(list_of_passes: List[str]):
    n_passes = len(list_of_passes)
    out = ""
    for i, pass_res in enumerate(list_of_passes):
        out += f"Pass {i+1}/{n_passes}:\n{pass_res}\n"
    return out

@click.command()
@click.option('--model_name', type=click.Choice(['openai', 'huggingface']), default="openai", help='The model to run.')
@click.option('--n_pass', type=int, default=1, help='Number of passes to run.')
@click.option('--n_logs_per_pass', type=int, default=5, help="Number of logs to process per pass.")
def main(model_name, n_pass, n_logs_per_pass):
    """
    Load the language model to use.
    """
    llm_interface = get_llm_interface(model_name)
    
    """
    Run the app.
    """
    app = Flask(__name__)

    @app.route("/", methods=["GET", "POST"])
    def index():
        question = ""
        display_answers = ""
        display_thinking = ""
        thinks = []
        answers = []
        logs_to_display = ""
        display_results = False

        question = request.form.get("question")
        if request.method == "POST":
            logs, thinks, answers = rag_query(question, llm_interface, n_logs_per_pass, n_pass)
            if n_pass > 1:
                logs = format_passes(logs)
                thinks = format_passes(thinks)
                answers = format_passes(answers)
            display_results = True

            logs_to_display = logs
            display_thinking = thinks
            display_answers = answers
    
        return render_template_string(
        """
        <form method="post">
            <label for="question">Ask {{ model_name }} a Question:</label>
            <textarea id="question" name="question" cols="40" rows="5" required></textarea>
            <input type="submit" value="Submit">
        </form>
        {% if display_results %}
            <h3>Retrieved Results:</h3>
            <pre style="white-space: pre-wrap;">{{ logs_to_display }}</pre>
            <h3>Model Name:</h3>
            <pre style="white-space: pre-wrap;">{{ model_name }}</pre>
            <h3>Question:</h3>
            <pre style="white-space: pre-wrap;">{{ question }}</pre>
            <h3>display_thinking:</h3>
            <pre style="white-space: pre-wrap;">{{ display_thinking }}</pre>
            <h3>Generated display_answers:</h3>
            <pre style="white-space: pre-wrap;">{{ display_answers }}</pre>
        {% endif %}
        """, question=question, 
            display_thinking=display_thinking,
            display_answers=display_answers, 
            logs_to_display=logs_to_display, 
            display_results=display_results, 
            model_name=llm_interface.model_name)
    
    app.run(debug=True)
if __name__ == "__main__":
    main()