FROM busybox

LABEL maintainer="Dita Pelaj <dita_s@live.com> and Luísa Hörlle de Castro <luisa@iservport.com>"
LABEL description="Activation Base for AI Project 'Appointment or Disappointment'"
LABEL course="M. Grum: Advanced AI-based Application Systems"
LABEL ownership="Created for the Junior Chair for Business Information Science, University of Potsdam"
LABEL data_origin="Your data scraping source"
LABEL license="AGPL-3.0"

RUN mkdir -p /tmp/activationBase

COPY data/cleaned/activation_data.csv /tmp/activationBase/
COPY images/activationBase_appointmentOrDisappointment/README.activationBase.md /tmp/activationBase/