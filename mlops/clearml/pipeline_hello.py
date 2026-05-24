from clearml import PipelineController

def hello_world():
    print("Hello world from ClearML agent")

pipe = PipelineController(
    name="Test Pipeline",
    project="mlops",
    version="1.0.0"
)

pipe.add_function_step(
    name="hello",
    function=hello_world,
    function_return=[],
    packages=[]
)

if __name__ == "__main__":
    pipe.start(queue="default")
    print("Pipeline submitted")