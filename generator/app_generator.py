from dotenv import load_dotenv
import os
import grpc
import asyncio

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import generator_pb2
import generator_pb2_grpc

load_dotenv()
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

class GeneratorServicer(generator_pb2_grpc.GeneratorServiceServicer):
    def __init__(self):
        self.llm = ChatOpenAI(
            model=LLM_MODEL,
            openai_api_key=OPENAI_API_KEY,
            temperature=0  
        )
        self.prompt = prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=(
            "너는 대학 소프트웨어융합대학(소융대)의 규정 및 교과과정 담당 조교야.\n"
            "주어진 자료(context) 내의 내용만 근거로 답변해.\n"
            "추측이나 일반적 지식으로 답하지 마. 근거가 없으면 '자료 내에서 확인 불가'라고 명시해.\n"
            "답변 마지막 줄에는 반드시 근거 출처를 [시트명(행번호)] 형태로 표기해.\n\n"
            "──────────────────────────────\n"
            "【자료】\n{context}\n"
            "──────────────────────────────\n"
            "【질문】\n{question}\n"
            "──────────────────────────────\n"
            "【답변】\n"
            "※ 참고:\n"
            "- 이 단과대의 공식 명칭은 '소프트웨어융합대학'이며, 줄여서 '소융대'라고 부른다.\n"
            "- 소융대에는 세 개의 학과가 있으며, 약칭은 다음과 같다:\n"
            "  • 컴퓨터공학과 → '컴공'\n"
            "  • 인공지능학과 → '인지' 또는 '인지과'\n"
            "  • 소프트웨어융합학과 → '소융'\n"
            "- 답변 시 '소융대'(단과대)와 '소융과'(학과)를 반드시 구분해서 사용해."

            ),
        )
        self.chain = self.prompt | self.llm | StrOutputParser()
    
    async def Generate(self, request, context):
        query = request.query
        contexts = request.contexts
        
        context_str = "\n\n".join(contexts)
        
        response_text = await self.chain.ainvoke({
                "context": context_str,
                "question": query
            })
               
        return generator_pb2.GenerateResponse(answer=response_text)
            
 
async def serve():
    server = grpc.aio.server()
    generator_pb2_grpc.add_GeneratorServiceServicer_to_server(GeneratorServicer(), server)
    server.add_insecure_port('[::]:50052')
    print("Generator gRPC Server running on port 50052")
    await server.start()
    await server.wait_for_termination()

if __name__ == "__main__":
    asyncio.run(serve())   


