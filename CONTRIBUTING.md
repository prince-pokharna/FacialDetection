# Contributing to BioScan 2.0

Thank you for contributing to the Face Recognition Attendance System!

## Development Workflow

1. Fork the repository and create a feature branch:
   ```bash
   git checkout -b feature/my-improvement
   ```

2. Set up local development environment:
   ```bash
   # Backend
   pip install -r backend/requirements.txt
   python ml/src/weights_manager.py

   # Frontend
   cd frontend
   npm install
   ```

3. Run verification before committing:
   ```bash
   # Backend tests & coverage
   pytest -v --cov=backend.app

   # Frontend build & unit tests
   cd frontend
   npm test
   npm run build
   ```

4. Code Style & Standards:
   - Backend: Ruff linting and type annotations.
   - Frontend: TypeScript strict type checks, Tailwind CSS formatting.
   - Commit messages: Follow Conventional Commits format (e.g. `feat: add landmark liveness challenge`).

5. Submit a Pull Request targeting the `main` branch with an explanation of your changes and test results.
