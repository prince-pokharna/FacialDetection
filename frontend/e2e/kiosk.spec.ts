import { test, expect } from '@playwright/test';

test.describe('Face Recognition Attendance Kiosk E2E', () => {
  test('renders landing page with 3D scene and navigation', async ({ page }) => {
    await page.goto('/');
    await expect(page).toHaveTitle(/BioScan/);
    await expect(page.getByRole('heading', { level: 1 })).toContainText('Facial');
  });

  test('navigates to live kiosk and loads webcam stream', async ({ page }) => {
    await page.goto('/');
    await page.getByRole('button', { name: /Live Kiosk/i }).first().click();
    await expect(page.getByRole('heading', { name: /Live Attendance Kiosk/i })).toBeVisible();
  });
});
