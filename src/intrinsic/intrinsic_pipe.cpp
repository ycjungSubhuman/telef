#include <pcl/io/image_rgb24.h>

#include "intrinsic/intrinsic_pipe.h"
#include "intrinsic/intrinsic.h"
#include "io/png.h"


namespace telef::intrinsic
{
boost::shared_ptr<PCANonRigidFittingResult>
IntrinsicPipe::_processData(boost::shared_ptr<PCANonRigidFittingResult> in)
{
  std::vector<uint8_t> rgb(in->image->getWidth()*in->image->getHeight()*3); 
  in->image->fillRaw(rgb.data());
  std::vector<uint16_t> depth = in->rendered_depth;
  std::vector<float> normal = in->rendered_normal;

  std::vector<double> intensity(rgb.size()/3);

  IntrinsicDecomposition dec;
  dec.initialize(
      rgb.data(), normal.data(), depth.data(),
      in->image->getWidth(), in->image->getHeight());
  dec.process(intensity.data());
  dec.release();

  std::vector<uint8_t> albedo(rgb.size());
  for(int i=0; i<rgb.size(); i++)
    {
      if(depth[i/3]==65535)
      {
        albedo[i] = 0;
        continue;
      }
      double tmp = static_cast<double>(rgb[i])/255.0;
      tmp/=intensity[i/3];
      if(tmp>1.0)
        tmp=1.0;
      // albedo[i] = static_cast<uint8_t>(static_cast<double>(rgb[i])/intensity[i/3]);
      albedo[i] = static_cast<uint8_t>(tmp*255.0);
    }

  pcl::io::FrameWrapper::Ptr wrapper = boost::make_shared<BufferFrameWrapper>(
      albedo, in->image->getWidth(), in->image->getHeight());
  in->original = in->image;
  in->image = boost::make_shared<pcl::io::ImageRGB24>(wrapper);
  in->intensity = intensity;

  return in;
}
}
