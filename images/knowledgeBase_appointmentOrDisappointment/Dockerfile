FROM busybox

LABEL maintainer="Dita Pelaj and Luísa Hörlle de Castro"
LABEL description="Knowledge Base for AI Project 'Appointment or Disappointment'"
LABEL course="M. Grum: Advanced AI-based Application Systems"
LABEL ownership="Dita Pelaj and Luísa Hörlle de Castro"
LABEL license="AGPL-3.0"

RUN mkdir -p /tmp/knowledgeBase

COPY code/model/currentAiSolution.h5 /tmp/knowledgeBase/
COPY images/knowledgeBase_appointmentOrDisappointment/README.knowledgeBase.md /tmp/knowledgeBase/

CMD ["sh"]
