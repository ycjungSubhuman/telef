#include "io/fetcher.h"

namespace telef::io {
    template<class T>
    void Fetcher<T>::callback(const T &fetchedInstance)
    {
        std::scoped_lock lock{this->dataMutex};
        onDataReady(fetchedInstance);
    }

    void CloudFetcher::onDataReady(const CloudFetcher::InstancePtr &fetchedInstance)
    {
        this->currentCloud = fetchedInstance;
    }
}
