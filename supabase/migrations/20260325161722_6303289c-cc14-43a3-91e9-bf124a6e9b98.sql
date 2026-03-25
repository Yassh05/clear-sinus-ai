
-- Create storage bucket for patient medical reports
INSERT INTO storage.buckets (id, name, public) VALUES ('medical-reports', 'medical-reports', true);

-- Allow anyone to upload files (no auth required for this app)
CREATE POLICY "Allow public uploads" ON storage.objects FOR INSERT TO anon, authenticated WITH CHECK (bucket_id = 'medical-reports');

-- Allow anyone to read files
CREATE POLICY "Allow public reads" ON storage.objects FOR SELECT TO anon, authenticated USING (bucket_id = 'medical-reports');

-- Allow anyone to delete their uploads
CREATE POLICY "Allow public deletes" ON storage.objects FOR DELETE TO anon, authenticated USING (bucket_id = 'medical-reports');
