from api import API

app = API()


def home(request, response):
    response.text = "Hello from the HOME page"


def about(request, response):
    response.text = "Hello from the ABOUT page"


@app.route("/hello/{name}")
def greeting(request, response, name):
    response.text = f"Hello, {name}"


@app.route("/tell/{age:d}")
def age(requst, response, age):
    response.text = f"Age is {age}"


@app.route("/sum/{num_1:d}/{num_2:d}")
def sum(request, response, num_1, num_2):
    total = int(num_1) + int(num_2)
    response.text = f"{num_1} + {num_2} = {total}"


@app.route("/book")
class BooksResource:
    def get(self, req, resp):
        resp.text = "Books Page"

    def post(self, req, resp):
        resp.text = "Endpoint to create a book"


def handler(req, resp):
    resp.text = "sample"


app.add_route("/sample", handler)


@app.route("/template")
def template_handler(req, resp):
    resp.body = app.template(
        "index.html", context={"name": "Bumbo", "title": "Best Framework"}
    ).encode()
