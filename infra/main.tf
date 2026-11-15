terraform {
  required_version = ">= 1.10.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 6.0"
    }
  }

  backend "s3" {
    key          = "weir/terraform.tfstate"
    encrypt      = true
    use_lockfile = true
  }
}

provider "aws" {
  region = var.region

  default_tags {
    tags = local.tags
  }
}

data "aws_caller_identity" "current" {}
data "aws_partition" "current" {}

data "aws_vpc" "default" {
  default = true
}

data "aws_subnets" "default" {
  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.default.id]
  }
}

locals {
  name       = "weir"
  account_id = data.aws_caller_identity.current.account_id
  partition  = data.aws_partition.current.partition
  bucket     = "weir-artifacts-${local.account_id}-${var.region}"
  image      = "${var.image_repository}:${var.image_tag}"
  vcpus      = "4"
  memory     = "8192"
  gpus       = "1"
  cpu_vcpus  = "16"
  cpu_memory = "30000"
  timeout    = 21600
  retries    = 5

  tags = {
    Project   = "weir"
    ManagedBy = "opentofu"
  }
}
