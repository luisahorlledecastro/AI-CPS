FROM busybox

LABEL maintainer="Dita Pelaj <dita_s@live.com> and Luísa Hörlle de Castro <luisa@iservport.com>"
LABEL description="Activation Base for AI Project 'Appointment or Disappointment'"
LABEL course="M. Grum: Advanced AI-based Application Systems"
LABEL data_origin=""
LABEL license="AGPL-3.0"

RUN mkdir -p /tmp/activationBase

COPY data/cleaned/activation_data.csv /tmp/activationBase/
COPY images/activationBase_appointmentOrDisappointment/README.activationBase.md /tmp/activationBase/

#  TODO add data origin