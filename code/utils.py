import SimpleITK as sitk



def convert_cuda(item):
    for key in item.keys():
        if key not in ['name', 'dst_name']:
            item[key] = item[key].float().cuda()
    return item


def save_nifti(image, path):
    out = sitk.GetImageFromArray(image)
    sitk.WriteImage(out, path)


def read_nifti(path):
    itk_img = sitk.ReadImage(path)
    image = sitk.GetArrayFromImage(itk_img)
    return image


def add_argument(parser, train=True):
    parser.add_argument('--name', type=str, default='baseline')
    parser.add_argument('--epoch', type=int, default=400)
    parser.add_argument('--dst_list', type=str, default='knee_zhao')
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--combine', type=str, default='mlp')
    parser.add_argument('--num_views', type=int, default=10)
    parser.add_argument('--view_offset', type=int, default=0)
    parser.add_argument('--out_res', type=int, default=256)
    parser.add_argument('--eval_npoint', type=int, default=100000)
    parser.add_argument('--visualize', action='store_true', default=False)
    
    if train:
        parser.add_argument('--batch_size', type=int, default=4)
        parser.add_argument('--lr', type=float, default=1e-2)
        parser.add_argument('--num_workers', type=int, default=2)
        parser.add_argument('--num_points', type=int, default=10000)
        parser.add_argument('--random_views', action='store_true', default=False)
    
    return parser
