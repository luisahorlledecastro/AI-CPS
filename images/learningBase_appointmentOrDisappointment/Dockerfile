# Dockerfile.learningBase
FROM busybox

LABEL maintainer="Dita Pelaj <dita_s@live.com> and Luísa Hörlle de Castro <luisa@iservport.com>"
LABEL description="Learning Base for AI Project 'Appointment or Disappointment'"
LABEL course="Advanced AI-based Application Systems"

RUN mkdir -p /tmp/learningBase/train /tmp/learningBase/validation

COPY data/cleaned/training_data.csv /tmp/learningBase/train/
COPY data/cleaned/test_data.csv /tmp/learningBase/validation/

COPY images/learningBase_appointmentOrDisappointment/README.learningBase.md /tmp/learningBase/

CMD ["sh"]


# note: image must be built two levels up at ../../