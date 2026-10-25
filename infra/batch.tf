resource "aws_security_group" "batch" {
  name_prefix = "${local.name}-batch-"
  description = "AWS Batch GPU workers; egress only"
  vpc_id      = data.aws_vpc.default.id

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  lifecycle {
    create_before_destroy = true
  }
}

resource "aws_cloudwatch_log_group" "batch" {
  name              = "/aws/batch/${local.name}"
  retention_in_days = 30
}

resource "aws_batch_compute_environment" "gpu" {
  name         = "${local.name}-gpu"
  type         = "MANAGED"
  state        = "ENABLED"
  service_role = aws_iam_role.batch_service.arn

  compute_resources {
    type                = "EC2"
    allocation_strategy = "BEST_FIT_PROGRESSIVE"
    min_vcpus           = 0
    max_vcpus           = var.max_vcpus
    instance_type       = [var.gpu_instance_type]
    subnets             = data.aws_subnets.default.ids
    security_group_ids  = [aws_security_group.batch.id]
    instance_role       = aws_iam_instance_profile.instance.arn

    ec2_configuration {
      image_type = "ECS_AL2023_NVIDIA"
    }
  }
}

resource "aws_batch_job_queue" "gpu" {
  name     = "${local.name}-gpu"
  state    = "ENABLED"
  priority = 1

  compute_environment_order {
    order               = 1
    compute_environment = aws_batch_compute_environment.gpu.arn
  }
}

resource "aws_batch_job_definition" "train" {
  name                  = "${local.name}-train"
  type                  = "container"
  platform_capabilities = ["EC2"]

  container_properties = jsonencode({
    image = local.image

    resourceRequirements = [
      { type = "VCPU", value = local.vcpus },
      { type = "MEMORY", value = local.memory },
      { type = "GPU", value = local.gpus },
    ]

    jobRoleArn       = aws_iam_role.job.arn
    executionRoleArn = aws_iam_role.execution.arn

    environment = [
      { name = "WEIR_RUN_DIR", value = "/work/run" },
      { name = "WEIR_REQUIRE_GPU", value = "1" },
      { name = "AWS_REGION", value = var.region },
      { name = "HYDRA_FULL_ERROR", value = "1" },
    ]

    logConfiguration = {
      logDriver = "awslogs"
      options = {
        "awslogs-group"         = aws_cloudwatch_log_group.batch.name
        "awslogs-region"        = var.region
        "awslogs-stream-prefix" = "train"
      }
    }
  })

  retry_strategy {
    attempts = local.retries
  }

  timeout {
    attempt_duration_seconds = local.timeout
  }
}
