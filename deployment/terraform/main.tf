provider "aws" {
  region = "us-east-1"
}

resource "aws_security_group" "vein_app_sg" {
  name        = "vein-app-sg"
  description = "Allow port 8501 for Streamlit"

  ingress {
    from_port   = 8501
    to_port     = 8501
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_instance" "app_server" {
  ami           = "ami-0c7217cdde317cfec" # Ubuntu 22.04 LTS (us-east-1)
  instance_type = "t2.micro"
  key_name      = "your-ssh-key-name" # Change this to your actual key name!
  security_groups = [aws_security_group.vein_app_sg.name]

  tags = {
    Name = "VeinAuthServer"
  }
}
