import shutil
from fastapi import FastAPI,UploadFile,File,Form
from fastapi.middleware.cors import CORSMiddleware
from prediction_script_complex import predict_ans_complex
from prediction_script_simple import predict_ans_simple
from prediction_script_satellite import predict_ans_satellite
app = FastAPI()

origins = ["http://localhost:3000",
    "localhost:3000"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/complex")
async def root(ques:str,file:UploadFile = File(...)):
    with open(f'images\\complex_image.jpg',"wb") as buffer:
       shutil.copyfileobj(file.file,buffer)
    answer = predict_ans_complex(ques,"images\\complex_image.jpg")
    return {"file_name":file.filename,"question":ques,"answer":answer}

@app.post("/complexlocal")
async def root(ques: str = Form(...),path: str = Form(...)):
    answer = predict_ans_complex(ques,path)
    return {"answer":answer}


@app.post("/simplelocal")
async def root(ques: str = Form(...),path: str = Form(...)):
    answer = predict_ans_simple(path,ques)
    return {"answer":answer}

@app.post("/satellitelocal")
async def root(ques: str = Form(...),path: str = Form(...)):
    answer = predict_ans_satellite(path,ques)
    return {"answer":answer}