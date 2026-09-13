import unittest
import numpy as np
from PIL import Image
from metrics import evaluate, load_reference
import app

class MetricsTests(unittest.TestCase):
    def test_known_confusion(self):
        p=np.array([[1,1],[0,0]],bool); t=np.array([[1,0],[1,0]],bool)
        m=evaluate(p,t)
        self.assertAlmostEqual(m['mIoU (foreground + background)'],1/3)
        self.assertEqual(m['Foreground Dice'],.5)
        self.assertEqual([m[k] for k in ['TP','FP','FN','TN']],[1,1,1,1])
    def test_empty_and_perfect(self):
        z=np.zeros((3,3),bool)
        m=evaluate(z,z)
        self.assertIsNone(m['Foreground IoU'])
        self.assertEqual(m['mIoU (foreground + background)'],1)
        self.assertIsNone(m['Foreground precision'])
        z[0,0]=True
        self.assertEqual(evaluate(z,z)['Foreground Dice'],1)
    def test_mask_validation(self):
        for scale in [1,255]:
            a=np.array([[0,scale],[scale,0]],np.uint8)
            np.testing.assert_array_equal(load_reference(Image.fromarray(a),(2,2)),a>0)
        with self.assertRaises(ValueError):
            load_reference(Image.new('RGB',(2,2),'red'),(2,2))
        with self.assertRaises(ValueError):
            load_reference(Image.new('L',(3,3)),(2,2))
        with self.assertRaises(ValueError):
            load_reference(Image.new('L',(2,2),128),(2,2))
    def test_cleanup_reversible(self):
        s=app.fresh(np.zeros((20,20,3),np.uint8))
        mask=np.zeros((20,20),bool); mask[3:12,3:12]=1; mask[18,18]=1
        s['raw_mask']=mask.copy(); s['mask']=mask.copy()
        s=app.clean_mask(s,2,0,False,False)[0]
        self.assertEqual(s['mask'].sum(),81)
        s=app.clean_mask(s,0,0,False,False)[0]
        np.testing.assert_array_equal(s['mask'],mask)
    def test_report_and_invalidation(self):
        s=app.fresh(np.zeros((2,2,3),np.uint8)); s['mask']=np.array([[1,0],[0,0]],bool)
        status,miou,summary,rows,errors,path=app.evaluate_mask(
            s, Image.fromarray(s['mask'].astype('uint8')*255)
        )
        self.assertAlmostEqual(miou, 100.0)
        self.assertIn('Measured binary mIoU', summary)
        self.assertIn('Measured mIoU: 100.00%', status)
        self.assertTrue(path.endswith('metrics.json'))
        self.assertEqual(errors.shape,(2,2,3))
        self.assertEqual(app.response(s,'changed')[-3:],(None,None,None))

if __name__=='__main__': unittest.main()
