FROM busybox

LABEL maintainer="Dita Pelaj and Luísa Hörlle de Castro"
LABEL description="Knowledge Base for AI Project 'Appointment or Disappointment'"
LABEL course="M. Grum: Advanced AI-based Application Systems"
LABEL ownership="Dita Pelaj and Luísa Hörlle de Castro"
LABEL license="AGPL-3.0"

RUN mkdir -p /knowledgeBase

COPY currentAiSolution.keras /knowledgeBase/
COPY ols_model.pkl /knowledgeBase
COPY scaler.pkl /knowledgeBase
COPY README.knowledgeBase.md /knowledgeBase/


# CMD ["sh"]
