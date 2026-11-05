# Infrastructure

This project contains the containerisation files used to package and run the Gen-AI repository services.

## Structure

```text
infrastructure/
├── docker/
│   └── analyst/       # Analyst service container configuration
├── Dockerfile         # Root container image definition
└── README.md
```

## Usage

Build the image from the repository root:

```bash
docker build -f infrastructure/Dockerfile -t gen-ai .
```

The `docker/analyst/` directory contains the service-specific Docker configuration. Review its files before starting a container because runtime environment variables and API credentials may be required.