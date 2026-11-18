# Biometric Privacy, Security & Compliance Policy

**Date**: November 2026  
**Applicable Regulations**: India's Digital Personal Data Protection Act (DPDP Act 2023), EU General Data Protection Regulation (GDPR), US State Biometric Privacy Acts (e.g., BIPA).

---

## 1. Classification of Biometric Data

Under international privacy frameworks, facial biometric information constitutes **Sensitive Personal Data / Special Category Data**. A facial biometric vector represents an intrinsic physiological characteristic that uniquely identifies an individual. Consequently, this system operates under strict data minimization, purpose limitation, and consent-driven storage principles.

---

## 2. Core Privacy Engineering Guarantees

### 2.1 No Raw Facial Images Stored by Default
- The system **does not persist raw facial photography** in production databases or object storage.
- Immediately upon capture during enrollment or kiosk scanning, face crops are processed in volatile memory through the ArcFace neural network to extract a 512-dimensional floating-point mathematical embedding.
- Raw pixel buffers are garbage-collected immediately following vector extraction.
- The 512-dimensional vector is irreversible—it cannot be reverse-engineered or reconstructed into the original photographic image of the human face.

### 2.2 Mandatory Explicit Consent
- In accordance with Section 6 of India's DPDP Act 2023 and Article 9(2)(a) of the EU GDPR, biometric data processing is prohibited without clear, affirmative, explicit consent.
- The enrollment wizard (`EnrollmentWizard.tsx` and `POST /people`) requires a mandatory consent checkbox before image submission.
- The consent event is cryptographically audited and permanently linked with an ISO 8601 UTC timestamp (`consent_timestamp`) alongside the individual's record.

### 2.3 Right to Erasure ("Right to be Forgotten")
- In accordance with DPDP Act Section 12 and GDPR Article 17, any enrolled individual possesses the unqualified right to have their biometric data permanently deleted.
- The administrative endpoint `DELETE /people/{id}` triggers an automated cascade:
  1. Removes person profile metadata from PostgreSQL/SQLite.
  2. Permanently deletes all associated 512-dimensional mathematical embeddings (`ondelete="CASCADE"`).
  3. Purges the in-memory vector index gallery in the active runtime inference engine.
  4. Deletes historical biometric match logs associated with the identity.

---

## 3. Dataset Licensing & VGGFace2 Restrictions

> [!IMPORTANT]
> The **VGGFace2** dataset was constructed by the Visual Geometry Group at the University of Oxford strictly for **academic and non-commercial research purposes**.

- **No VGGFace2 Images in Production**: Raw VGGFace2 images or subsets are excluded from application Docker containers, git commits, and cloud deployment bundles via root `.gitignore`.
- **Pretrained Weights**: Production deployments utilize publicly released ArcFace/SFace backbones trained on ethically collected, permissible face recognition benchmarks (e.g., MS1MV2, Glint360k, or OpenCV Zoo CC-BY).
- **Benchmark Only**: In this repository, the legacy 25-identity subset is archived exclusively as a historic baseline benchmark in `legacy/` and is never bundled in public deployment artifacts.

---

## 4. Security Architecture & Threat Mitigation

### 4.1 HTTPS & Secure Context Enforcement
- Modern web browsers strictly mandate a Secure Context (HTTPS or `localhost`) for the `navigator.mediaDevices.getUserMedia` API.
- Production deployments must terminate TLS 1.3 at reverse proxies (Nginx / Cloudflare / Vercel) with HTTP Strict Transport Security (`HSTS`).

### 4.2 Presentation Attack Detection (Anti-Spoofing)
- To prevent impersonation via printed photographs or digital displays, all kiosk frames undergo passive liveness inspection (Fourier high-frequency moiré analysis, skin chrominance gamut verification, and specular reflection checks) before biometric vector comparison.

### 4.3 Credential Security & Secret Isolation
- Administrative and instructor passwords are hashed with `bcrypt` (work factor 12) with salt.
- Tokens are issued as cryptographically signed JSON Web Tokens (`HS256` or `RS256`) with strict expiration limits.
- Zero secrets are hardcoded in application source code; configuration is injected dynamically via `.env` files adhering to 12-Factor App standards.
- Cross-Origin Resource Sharing (`CORS`) is explicitly restricted to verified production domains via environment variables.

---

## 5. Security Headers Implementation

Production gateways must enforce the following HTTP response headers:
```http
Strict-Transport-Security: max-age=31536000; includeSubDomains; preload
X-Content-Type-Options: nosniff
X-Frame-Options: DENY
Content-Security-Policy: default-src 'self'; img-src 'self' data: blob:; connect-src 'self' ws: wss:;
Referrer-Policy: strict-origin-when-cross-origin
```
