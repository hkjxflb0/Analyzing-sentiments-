from fastapi import FastAPI, Form,Request
from fastapi.responses import HTMLResponse
from starlette.templating import Jinja2Templates

from models.vader_model import VADERModel
from models.Robert import TransformerModel
from models.toxicity_model import ToxicityModel
from ensemble.combine import EnsembleModel
from fastapi.staticfiles import StaticFiles

app = FastAPI()

# Initialize models
vader_model = VADERModel()
transformer_model = TransformerModel()
toxicity_model = ToxicityModel()
ensemble_model = EnsembleModel(vader_model, transformer_model, toxicity_model)

# Mount static files for templates
templates = Jinja2Templates(directory="templates")

def categorize_toxicity(toxicity_score: float) -> str:
    if toxicity_score < 0.2:
        return f"High Toxicity,{toxicity_score}"
    elif toxicity_score < 0.5:
        return f"Moderate Toxicity, {toxicity_score}"
    else:
        return f"Low Toxicity, {toxicity_score}"

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """
    Render the HTML page with the form.
    """
    return templates.TemplateResponse("index.html", {"request": request})

# Process form submission and return the analysis result
@app.post("/analyze/")
async def analyze(request: Request, text: str = Form(...)):
    """
    Analyze the sentiment and toxicity of the input text and render the result.
    """
    result = ensemble_model.predict(text)
    print(result)
    toxicity_category = categorize_toxicity(result['toxicity'])

    return templates.TemplateResponse("index.html", {
        "request": request,
        "result": result,
        "input_text": text,
        "toxicity_category": toxicity_category,
    })