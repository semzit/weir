variable "region" {
  type        = string
  description = "AWS region for the Batch stack."
}

variable "image_repository" {
  type        = string
  description = "Container image repository without the tag."
  default     = "ghcr.io/semzit/weir"
}

variable "image_tag" {
  type        = string
  description = "Immutable image tag, normally the full git SHA."
}

variable "gpu_instance_type" {
  type        = string
  description = "GPU instance type used by the Batch compute environment."
  default     = "g5.xlarge"
}

variable "max_vcpus" {
  type        = number
  description = "Maximum vCPUs the GPU compute environment may scale to."
  default     = 8

  validation {
    condition     = var.max_vcpus >= 4
    error_message = "max_vcpus must be at least 4 so one training job can run."
  }
}

variable "budget_email" {
  type        = string
  description = "Email address that receives the monthly budget alerts."
}

variable "monthly_budget_usd" {
  type        = number
  description = "Monthly EC2 budget limit in US dollars."
  default     = 50

  validation {
    condition     = var.monthly_budget_usd > 0
    error_message = "monthly_budget_usd must be positive."
  }
}
