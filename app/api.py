import litserve as ls


class RAGAPI(ls.LitAPI):

    def setup(self, device):
        from app.rag_pipeline import initialize_rag
        self.engine = initialize_rag()

    def decode_request(self, request):
        messages = getattr(request, "messages", [])

        for m in reversed(messages):
            if m.role == "user":
                return m.content

        return ""

    def predict(self, user_msg):
        ## TODO: Split the pipeline to allow different response modes
        response = self.engine.query(user_msg)
        
        for token in response.response_gen:
            print(token, end="", flush=True)
            yield str(token)