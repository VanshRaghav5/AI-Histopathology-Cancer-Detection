import axios from 'axios';

const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:8000';

export interface PredictionResponse {
    prediction: string;
    confidence: number;
    probabilities: {
        benign: number;
        malignant: number;
    };
    file_info: {
        filename: string;
        content_type: string;
        size: number;
    };
}

export const histopathologyAPI = {
    /**
     * Sends an image to the backend for cancer detection analysis
     * @param file - The image file to analyze
     * @returns Promise with prediction results
     */
    async predictCancer(file: File): Promise<PredictionResponse> {
        try {
            const formData = new FormData();
            formData.append('file', file);

            const response = await axios.post<PredictionResponse>(
                `${BACKEND_URL}/predict`,
                formData,
                {
                    headers: {
                        'Content-Type': 'multipart/form-data',
                    },
                }
            );

            return response.data;
        } catch (error) {
            console.error('Error during cancer prediction:', error);
            throw error;
        }
    },

    /**
     * Checks if the backend server is running
     * @returns Promise<boolean>
     */
    async checkHealth(): Promise<boolean> {
        try {
            const response = await axios.get(`${BACKEND_URL}/`);
            return response.data.status === 'running';
        } catch (error) {
            console.error('Error checking backend health:', error);
            return false;
        }
    },
};